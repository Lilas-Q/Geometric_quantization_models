"""Joint observed-history planning with causal state-conditioned field corrections.

The v9 backbone, raw-u field convention and two half-frame image updates remain
unchanged. Only the extra source/transport readouts are zero initialized.
"""
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import PhysicalBackbone
from .frozen_expm import frozen_exponential_step
from .transport import velocity_total_variation
from .memory import append_state
from .planning import FuturePlanner, StateCorrection

MODEL_ID = 'lhfm_moving_mnist_physical_fast_plan_v17_wide'

class Forecast(NamedTuple):
    prediction: torch.Tensor
    source_energy: torch.Tensor
    velocity_tv: torch.Tensor
    r_rms: torch.Tensor
    u_rms: torch.Tensor
    planned_coarse: torch.Tensor | None = None


class Prepared(NamedTuple):
    z0: torch.Tensor
    base_controls: torch.Tensor
    spatial_logits: torch.Tensor
    offset: torch.Tensor
    anchor: torch.Tensor
    plan: torch.Tensor

class PhysicalFastPlanMovingMNIST(PhysicalBackbone):
    def __init__(self, *, plan_channels=128, plan_depth=4, plan_expansion=4,
                 correction_hidden=192, **backbone_kwargs):
        super().__init__(**backbone_kwargs)
        sizes=(plan_channels,plan_depth,plan_expansion,correction_hidden)
        if any(type(n) is not int or n<1 for n in sizes):
            raise ValueError('positive integer planner dimensions required')
        self.plan_channels,self.plan_depth,self.plan_expansion,self.correction_hidden=sizes
        self.planner=FuturePlanner(self.latent_channels,plan_channels,plan_depth,plan_expansion)
        self.correction=StateCorrection(self.latent_channels,plan_channels,correction_hidden)
        # Shared across all ten plan features. Nonzero initialization gives the
        # planner direct gradients before the physical correction heads activate.
        self.plan_head=nn.Linear(plan_channels,1)
        nn.init.normal_(self.plan_head.weight,std=.02)
        nn.init.zeros_(self.plan_head.bias)
        self.float()
        if self.channels_last:
            for module in self.planner.modules():
                if isinstance(module,nn.Conv2d):module.to(memory_format=torch.channels_last)

    def config(self):
        return dict(super().config(),model_id=MODEL_ID,plan_channels=self.plan_channels,
            plan_depth=self.plan_depth,plan_expansion=self.plan_expansion,
            correction_hidden=self.correction_hidden,
            planner='one_observed_z0_to_ten_joint_future_features',
            planner_cadence='once_per_clip_with_shared_autograd_tape',
            correction='zero_initialized_additive_raw_r_u_heads_from_current_z_and_plan_k',
            generated_state_feedback=True,architecture_baseline='physical_multiscale_memory_v9',
            initialization='scratch_no_parent_weights',
            memory_projection='packed_qkv_cached_query_without_detach',
            frame_execution='fused_memory_recall_and_controls',
            terminal_state_update='omitted_unused_final_half_frame_state',
            planning_supervision='shared_linear_coarse_image_head_training_only',
            planning_target='avg_pool2d_future_frames_kernel4_stride4',
            planning_output_activation='identity',planning_future_weights='equal')

    def parameter_counts(self):
        counts=super().parameter_counts()
        counts.update(joint_planner=sum(p.numel() for p in self.planner.parameters()),
            state_correction=sum(p.numel() for p in self.correction.parameters()),
            training_only_plan_head=sum(p.numel() for p in self.plan_head.parameters()))
        return counts

    def prepare(self,context):
        p=super().prepare(context)
        return Prepared(*p,self.planner(p.z0))

    def predict_plan(self,plan):
        # One vectorized readout, using the SAME attached plan as the r/u branch.
        return self.plan_head(plan).permute(0,1,4,2,3).float()

    def fields(self,z,offset,plan):
        packed=torch.cat((self.source_head(z),self.transport_head(z)),-1).float()+offset
        packed=packed+self.correction(z,plan).float()
        field=F.pixel_shuffle(packed.permute(0,3,1,2).contiguous(),4)
        return field[:,:1],field[:,1:]

    def begin_frame(self,z,query,keys,values,image,history_z,base_controls,spatial_logits,anchor):
        z=self.memory(z,query,keys,values)
        controls=self.frame_controls(z,image,history_z,base_controls,spatial_logits,anchor)
        return z,controls

    def rollout(self,context,horizon=10,*,prepare_fn=None,fields_fn=None,
                state_fn=None,transport_fn=None,memory_encode_fn=None,frame_fn=None,
                supervise_plan=False,plan_head_fn=None):
        self.validate_context(context)
        if type(horizon) is not int or not 1<=horizon<=10:
            raise ValueError('horizon must be from one to ten')
        prepare_fn=self.prepare if prepare_fn is None else prepare_fn
        fields_fn=self.fields if fields_fn is None else fields_fn
        state_fn=self.update_state if state_fn is None else state_fn
        transport_fn=frozen_exponential_step if transport_fn is None else transport_fn
        memory_encode_fn=self.memory.encode if memory_encode_fn is None else memory_encode_fn
        frame_fn=self.begin_frame if frame_fn is None else frame_fn
        p=prepare_fn(context)
        coarse=None
        if supervise_plan:
            plan_head_fn=self.predict_plan if plan_head_fn is None else plan_head_fn
            coarse=plan_head_fn(p.plan[:,:horizon])
        image,z=context[:,-1].float(),p.z0
        query,key,value=memory_encode_fn(z)
        # Initial padding repeats the encoded observed-history state. It does
        # not invent future frames; all subsequent entries are generated states.
        keys=torch.stack([key]*self.memory.slots,-2)
        values=torch.stack([value]*self.memory.slots,-2)
        images,energies,variations,squares=[],[],[],[]
        for frame in range(horizon):
            z,controls=frame_fn(z,query,keys,values,image,p.z0,p.base_controls,p.spatial_logits,p.anchor)
            # Reuse attached controls, not detached controls, for both half frames.
            for substep in range(self.substeps):
                r,u=fields_fn(z,p.offset,p.plan[:,frame])
                energies.append(r.square().mean())
                variations.append(velocity_total_variation(u))
                squares.append(u.square().mean().detach())
                image=transport_fn(image,r,u,1./self.substeps)
                # The final image and every regularizer have already been
                # computed. This last z would have no downstream consumer.
                if frame+1<horizon or substep+1<self.substeps:
                    z=state_fn(z,image,controls.rotation,controls.spatial_weights,
                               controls.feedback_scale,controls.drift,p.anchor)
            images.append(image)
            if frame+1<horizon:
                query,key,value=memory_encode_fn(z)
                keys,values=append_state(keys,values,key,value)
        energy=torch.stack(energies).mean()
        return Forecast(torch.stack(images,1),energy,torch.stack(variations).mean(),
                        energy.detach().sqrt(),torch.stack(squares).mean().sqrt(),coarse)

    def forward(self,context,horizon=10,*,supervise_plan=False):
        return self.rollout(context,horizon,supervise_plan=supervise_plan)
