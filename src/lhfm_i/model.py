"""LHFM-I: a shared U-Net with transport and source outputs."""

from typing import NamedTuple
import math
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import Attention, ImageVelocityUNet, zero
from .geometry import CosineSpatialDerivative


class Fields(NamedTuple):
    velocity: torch.Tensor
    transport: torch.Tensor
    source: torch.Tensor


class ContiguousAttention(nn.Module):
    """Original attention arithmetic with contiguous head dimensions."""

    def __init__(self, original):
        super().__init__()
        self.heads = original.heads
        self.norm, self.qkv, self.out = original.norm, original.qkv, original.out

    def forward(self, x):
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)

        def arrange(value):
            return value.reshape(batch, self.heads, channels // self.heads,
                                 height * width).transpose(-1, -2).contiguous()

        hidden = F.scaled_dot_product_attention(arrange(q), arrange(k), arrange(v), dropout_p=0.)
        hidden = hidden.transpose(-1, -2).reshape(batch, channels, height, width)
        return (x + self.out(hidden)) * (2 ** -.5)


class LHFM_I(ImageVelocityUNet):
    """RGB image velocity v = r - (DF_J restricted to the grid) u.

    ``forward(image, time)`` returns v; ``fields(image, time)`` returns (v,u,r).
    u is the grid representation of the transport vector field. Its first
    component acts along image width (x), and its second along height (y).
    """

    def __init__(self, *, transport_scale=.125, **kwargs):
        if not math.isfinite(transport_scale) or transport_scale <= 0:
            raise ValueError("transport_scale must be finite and positive")
        super().__init__(**kwargs)
        # Preserve initialization and state-dict layout of the paper model.
        rng_state = torch.get_rng_state()
        head = nn.Conv2d(self.output.in_channels, 5, self.output.kernel_size,
                         padding=self.output.padding)
        torch.set_rng_state(rng_state)
        self.output = zero(head)
        self.operators = CosineSpatialDerivative(self.image_size)
        self.transport_scale = float(transport_scale)
        for module in list(self.modules()):
            for name, child in list(module.named_children()):
                if isinstance(child, Attention):
                    setattr(module, name, ContiguousAttention(child))

    def fields(self, image, time):
        raw = super().forward(image, time)
        dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=image.device.type, enabled=False):
            raw = raw.to(dtype)
            source = raw[:, 2:]
            transport = self.transport_scale * time.to(dtype)[:, None, None, None] * torch.tanh(raw[:, :2])
            dx, dy = self.operators.spatial_derivatives(image)
            velocity = torch.addcmul(source, dx, transport[:, 0:1], value=-1)
            velocity = torch.addcmul(velocity, dy, transport[:, 1:2], value=-1)
        return Fields(velocity, transport, source)

    def forward(self, image, time):
        return self.fields(image, time).velocity


def build_model(config):
    """Build LHFM-I from the public configuration dictionary."""
    return LHFM_I(**config["model"])
