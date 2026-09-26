"""Causal motion recall with a second-order spatial frozen-field image ODE.

Controls refresh once per predicted video frame, fields once per half frame.
Neither the learned field readout nor the full coupled state is globally linear.
"""
from __future__ import annotations
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F
from .context import HistoryLatentEncoder
from .frozen_expm import frozen_exponential_step, TERMS, MAX_SCALED_SHIFT
from .transport import velocity_total_variation
from .memory import MotionMemory, append_state

MODEL_ID = 'lhfm_moving_mnist_physical_fast_plan_v17_wide'


class Prepared(NamedTuple):
    z0: torch.Tensor
    base_controls: torch.Tensor
    spatial_logits: torch.Tensor
    offset: torch.Tensor
    anchor: torch.Tensor


class FrameControls(NamedTuple):
    rotation: torch.Tensor
    spatial_weights: torch.Tensor
    feedback_scale: torch.Tensor
    drift: torch.Tensor


class Forecast(NamedTuple):
    prediction: torch.Tensor
    source_energy: torch.Tensor
    velocity_tv: torch.Tensor
    r_rms: torch.Tensor
    u_rms: torch.Tensor


class PhysicalBackbone(nn.Module):
    def __init__(self, widths=(64,128,448), context_frames=10, substeps=2,
                 channels_last=True, latent_channels=384, depth=3, expansion=4,
                 spatial_groups=4, controller_hidden=256, field_hidden=192, mixing_rank=128,
                 memory_frames=6, memory_key_dim=128, motion_depth=2, motion_expansion=4,
                 motion_channels=320, memory_value_dim=256, memory_heads=4,
                 coarse_channels=192, coarse_depth=2):
        super().__init__()
        if type(context_frames) is not int or context_frames < 2:
            raise ValueError('at least two history frames required')
        if type(substeps) is not int or substeps != 2:
            raise ValueError('exactly two half-frame intervals required')
        if type(latent_channels) is not int or latent_channels < 4 or latent_channels % 2:
            raise ValueError('latent channels must be positive and even')
        if any(type(n) is not int or n < 1 for n in
               (depth,expansion,spatial_groups,controller_hidden,field_hidden,mixing_rank,
                memory_frames,memory_key_dim,motion_depth,motion_expansion,motion_channels,
                memory_value_dim,memory_heads,coarse_channels,coarse_depth)):
            raise ValueError('positive integer architectural sizes required')
        if latent_channels % spatial_groups or (latent_channels//spatial_groups) % 2:
            raise ValueError('even latent channels per spatial group required')
        if mixing_rank > latent_channels:
            raise ValueError('mixing rank cannot exceed latent channels')
        if motion_channels > latent_channels:
            raise ValueError('motion bottleneck cannot exceed latent channels')
        if memory_key_dim % memory_heads or memory_value_dim % memory_heads:
            raise ValueError('key and value dimensions must be divisible by heads')
        if memory_value_dim > latent_channels or coarse_channels > motion_channels:
            raise ValueError('memory and coarse projections must remain bottlenecks')
        if type(channels_last) is not bool:
            raise ValueError('channels_last must be boolean')
        if len(widths) != 3 or any(type(w) is not int or w < 4 for w in widths):
            raise ValueError('three widths >=4 required')
        self.widths, self.context_frames, self.substeps = tuple(widths),context_frames,substeps
        self.channels_last, self.latent_channels = channels_last,latent_channels
        self.depth,self.expansion,self.spatial_groups = depth,expansion,spatial_groups
        self.controller_hidden,self.field_hidden,self.mixing_rank = controller_hidden,field_hidden,mixing_rank
        self.memory_frames,self.memory_key_dim = memory_frames,memory_key_dim
        self.motion_depth,self.motion_expansion = motion_depth,motion_expansion
        self.motion_channels = motion_channels
        self.memory_value_dim,self.memory_heads = memory_value_dim,memory_heads
        self.coarse_channels,self.coarse_depth = coarse_channels,coarse_depth
        self.history = HistoryLatentEncoder(context_frames,widths,latent_channels,depth,expansion,
                                            channels_last,spatial_groups)
        self.memory = MotionMemory(latent_channels,memory_frames,memory_key_dim,motion_depth,
            motion_expansion,motion_channels,memory_value_dim,memory_heads,coarse_channels,coarse_depth)
        # Full z and history z0 plus the 16 current-minus-initial pixel phases.
        self.controller = nn.Sequential(nn.LayerNorm(2*latent_channels+16),
            nn.Linear(2*latent_channels+16,controller_hidden),nn.SiLU(),
            nn.Linear(controller_hidden,3*latent_channels+5*spatial_groups))
        # Source and raw transport have independent lightweight nonlinear heads.
        self.source_head = nn.Sequential(nn.Linear(latent_channels,field_hidden),nn.SiLU(),
                                         nn.Linear(field_hidden,16))
        self.transport_head = nn.Sequential(nn.Linear(latent_channels,field_hidden),nn.SiLU(),
                                            nn.Linear(field_hidden,32))
        self.feedback = nn.Linear(16,latent_channels,bias=False)
        # Low-rank channel exchange; no recurrent large dense convolution.
        self.channel_down = nn.Linear(latent_channels,mixing_rank,bias=False)
        self.channel_up = nn.Linear(mixing_rank,latent_channels,bias=False)
        nn.init.normal_(self.controller[-1].weight,std=.001)
        nn.init.zeros_(self.controller[-1].bias)
        for head in (self.source_head,self.transport_head):
            nn.init.normal_(head[-1].weight,std=.002)
            nn.init.zeros_(head[-1].bias)
        nn.init.normal_(self.feedback.weight,std=.01)
        nn.init.normal_(self.channel_down.weight,std=.02)
        nn.init.normal_(self.channel_up.weight,std=.002)
        self.float()
        if channels_last:
            for module in self.modules():
                if isinstance(module,nn.Conv2d):
                    module.to(memory_format=torch.channels_last)

    def config(self):
        return dict(model_id=MODEL_ID,widths=list(self.widths),context_frames=self.context_frames,
            substeps=self.substeps,channels_last=self.channels_last,latent_channels=self.latent_channels,
            depth=self.depth,expansion=self.expansion,spatial_groups=self.spatial_groups,
            controller_hidden=self.controller_hidden,field_hidden=self.field_hidden,mixing_rank=self.mixing_rank,
            memory_frames=self.memory_frames,memory_key_dim=self.memory_key_dim,
            motion_depth=self.motion_depth,motion_expansion=self.motion_expansion,
            motion_channels=self.motion_channels,
            memory_value_dim=self.memory_value_dim,memory_heads=self.memory_heads,
            coarse_channels=self.coarse_channels,coarse_depth=self.coarse_depth,
            temporal_memory='permanent_observed_history_anchor_and_recent_causal_states_cached_compressed_kv',
            spatial_motion_refinement='fine_H4_and_coarse_H8_depthwise5x5_residuals_once_per_frame',
            motion_update_cadence='once_per_video_frame_before_controls',
            latent_spatial_downsample=4,history_encoder='ordered_intensity_difference_coordinates_once',
            field_backbone='separate_source_transport_mlp_16_phase_pixel_shuffle',u_gate='none',
            state_transition='low_rank_channel_exchange_grouped_five_point_mixing_damped_rotations',
            channel_exchange_scale=.1,
            controls_update_cadence='once_per_video_frame_from_z_generated_J_and_history',
            controls_frozen_for_substeps=2,state_update_cadence='each_half_frame',field_calls_per_frame=2,
            state_feedback='condition_scaled_linear_map_of_pixel_unshuffle_J_minus_I10',
            spatial_units='pixels',time_units='video_frames',intensity_units='uint8_div_255',
            integrator='frozen_affine_signed_shifted_exponential_series',
            spatial_operator='zero_exterior_second_order_upwind_generator',
            matrix_storage='implicit_sparse_stencil',exponential_order=TERMS,
            max_scaled_shift=MAX_SCALED_SHIFT,shift_multiplier=1.5)

    def parameter_counts(self):
        count=lambda module:sum(p.numel() for p in module.parameters())
        return dict(history_encoder_and_controls=count(self.history),adaptive_controller=count(self.controller),
                    temporal_memory_and_spatial_refinement=count(self.memory),
                    source_head=count(self.source_head),transport_head=count(self.transport_head),
                    linear_image_feedback=count(self.feedback),
                    low_rank_channel_exchange=count(self.channel_down)+count(self.channel_up),total=count(self))

    def validate_context(self,context):
        if (context.ndim!=5 or context.shape[1:3]!=(self.context_frames,1)
            or min(context.shape[-2:])<16 or any(d%4 for d in context.shape[-2:])
            or not context.is_floating_point()):
            raise ValueError('expected floating [B,T,1,H,W], H/W >=16 and divisible by4')

    def pack_image(self,image):
        return F.pixel_unshuffle(image.float(),4).permute(0,2,3,1).contiguous()

    def prepare(self,context):
        z0,controls,spatial_logits,offset=self.history(context)
        return Prepared(z0.float(),controls.float(),spatial_logits.float(),offset.float(),
                        self.pack_image(context[:,-1]))

    def frame_controls(self,z,image,history_z,base_controls,spatial_logits,anchor):
        value=torch.cat((z,history_z,self.pack_image(image)-anchor),-1)
        correction=self.controller(value).float()
        d=self.latent_channels
        raw=base_controls+correction[...,:3*d]
        logits=spatial_logits+correction[...,3*d:].reshape(*z.shape[:-1],self.spatial_groups,5)
        # Neural coefficient prediction uses autocast; physical coefficients stay FP32.
        with torch.autocast(image.device.type,enabled=False):
            omega,log_decay,feedback,drift=torch.split(raw.float(),(d//2,d//2,d,d),-1)
            dt=1./self.substeps
            rho=torch.exp(-dt*F.softplus(log_decay))
            angle=dt*omega
            rotation=torch.stack((rho*angle.cos(),rho*angle.sin()),-1)
            weights=logits.float().softmax(-1)
            return FrameControls(rotation,weights,feedback.sigmoid(),dt*drift)

    def fields(self,z,offset):
        packed=torch.cat((self.source_head(z),self.transport_head(z)),-1).float()+offset
        field=F.pixel_shuffle(packed.permute(0,3,1,2).contiguous(),4)
        return field[:,:1],field[:,1:]

    def mix_state(self,z,weights):
        shape=z.shape
        grouped=z.reshape(*shape[:-1],self.spatial_groups,self.latent_channels//self.spatial_groups)
        left=torch.cat((grouped[:,:,:1],grouped[:,:,:-1]),2)
        right=torch.cat((grouped[:,:,1:],grouped[:,:,-1:]),2)
        up=torch.cat((grouped[:,:1],grouped[:,:-1]),1)
        down=torch.cat((grouped[:,1:],grouped[:,-1:]),1)
        mixed=(weights[...,:1]*grouped + weights[...,1:2]*left + weights[...,2:3]*right
               + weights[...,3:4]*up + weights[...,4:5]*down)
        return mixed.reshape(shape)

    def update_state(self,z,image,rotation,spatial_weights,feedback_scale,drift,anchor):
        with torch.autocast(image.device.type,enabled=False):
            z=z.float()
            exchanged=z+.1*self.channel_up(self.channel_down(z))
            mixed=self.mix_state(exchanged,spatial_weights)
            pairs=mixed.reshape(*mixed.shape[:-1],self.latent_channels//2,2)
            c,s=rotation[...,0],rotation[...,1]
            x,y=pairs[...,0],pairs[...,1]
            moved=torch.stack((c*x-s*y,s*x+c*y),-1).flatten(-2)
            feedback=self.feedback(self.pack_image(image)-anchor)
            return moved+feedback_scale*feedback+drift

