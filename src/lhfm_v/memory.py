"""Anchored causal recall and two-scale spatial motion refinement.

Only observed-history and completed predicted states enter the cached K/V bank.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .context import norm


class SpatialResidual(nn.Module):
    def __init__(self, channels, expansion, kernel_size=5):
        super().__init__()
        self.norm = norm(channels)
        self.depthwise = nn.Conv2d(channels, channels, kernel_size,
                                  padding=kernel_size//2, groups=channels)
        self.expand = nn.Conv2d(channels, expansion*channels, 1)
        self.project = nn.Conv2d(expansion*channels, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), .01))

    def forward(self, value):
        update = self.depthwise(self.norm(value))
        update = self.project(torch.nn.functional.silu(self.expand(update)))
        return value.float() + self.scale * update.float()


class MotionMemory(nn.Module):
    def __init__(self, channels, frames=6, key_dim=64, depth=2, expansion=4,
                 spatial_channels=192, value_dim=128, heads=4,
                 coarse_channels=128, coarse_depth=2):
        super().__init__()
        if key_dim % heads or value_dim % heads:
            raise ValueError('key and value dimensions must be divisible by heads')
        self.frames, self.key_dim, self.value_dim, self.heads = frames, key_dim, value_dim, heads
        self.slots = frames + 1
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 2*key_dim+value_dim, bias=False)
        self.output = nn.Linear(value_dim, channels, bias=False)
        self.age_bias = nn.Parameter(torch.zeros(heads, self.slots))
        self.spatial_down = nn.Linear(channels, spatial_channels, bias=False)
        self.fine = nn.Sequential(*[SpatialResidual(spatial_channels, expansion) for _ in range(depth)])
        self.coarse_down = nn.Conv2d(spatial_channels, coarse_channels, 1, bias=False)
        self.coarse = nn.Sequential(*[SpatialResidual(coarse_channels, expansion)
                                      for _ in range(coarse_depth)])
        self.coarse_up = nn.Conv2d(coarse_channels, spatial_channels, 1, bias=False)
        self.spatial_up = nn.Linear(spatial_channels, channels, bias=False)

    def encode(self, state):
        # This completed state is also the next frame's query. No detach:
        # all three projections retain their gradients back to this state.
        return self.qkv(self.norm(state)).split((self.key_dim,self.key_dim,self.value_dim),-1)

    def forward(self, state, query, keys, values):
        # Temporal attention per spatial location, not dense space-time attention.
        query = query.unflatten(-1, (self.heads, self.key_dim//self.heads))
        keys = keys.unflatten(-1, (self.heads, self.key_dim//self.heads)).transpose(-3, -2)
        values = values.unflatten(-1, (self.heads, self.value_dim//self.heads)).transpose(-3, -2)
        scores = (query.unsqueeze(-2).float()*keys.float()).sum(-1) / math.sqrt(self.key_dim//self.heads)
        weights = (scores+self.age_bias).softmax(-1)
        recalled = (weights.unsqueeze(-1)*values.float()).sum(-2).flatten(-2)
        updated = state.float() + .1*self.output(recalled).float()
        reduced = self.spatial_down(updated)
        spatial = reduced.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        fine_delta = self.fine(spatial)-spatial.float()
        coarse = self.coarse_down(F.avg_pool2d(spatial, 2))
        coarse_delta = self.coarse_up(self.coarse(coarse)-coarse.float())
        # Resample latent features only. Image generation retains the same ODE.
        coarse_delta = F.interpolate(coarse_delta.float(), size=spatial.shape[-2:],
                                     mode='bilinear', align_corners=False)
        delta = (fine_delta+coarse_delta).permute(0, 2, 3, 1).contiguous()
        return updated + self.spatial_up(delta).float()


def append_state(keys, values, key, value):
    """Keep the observed-history anchor; append attached states without tape mutation."""
    return (torch.cat((keys[..., :1, :], keys[..., 2:, :], key.unsqueeze(-2)), -2),
            torch.cat((values[..., :1, :], values[..., 2:, :], value.unsqueeze(-2)), -2))
