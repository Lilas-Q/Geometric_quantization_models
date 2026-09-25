"""Full-band cosine spatial derivatives on a pixel-centered grid."""

from __future__ import annotations
import math
import torch
from torch import nn

def validate_images(images: torch.Tensor, channels: int = 3) -> None:
    if (images.ndim != 4 or images.shape[1] != channels or min(images.shape) <= 0
            or not images.is_floating_point()):
        raise ValueError("images must be nonempty floating [B,C,H,W]")


def cosine_basis(coordinate: torch.Tensor, size: int, derivative: bool = False):
    modes = torch.arange(size, device=coordinate.device, dtype=coordinate.dtype)
    norm = torch.full_like(modes, math.sqrt(2 / size))
    norm[0] = 1 / math.sqrt(size)
    argument = math.pi * coordinate[..., None] * modes
    return ((-math.pi * modes * argument.sin()) if derivative else argument.cos()) * norm


class CosineSpatialDerivative(nn.Module):
    """Spatial Jacobian of the full-band cosine extension, without filtering."""

    def __init__(self, image_size: int = 32):
        super().__init__()
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1:
            raise ValueError("invalid cosine operator dimensions")
        self.image_size = image_size
        self._register_fixed_matrices(torch.device("cpu"))

    def _register_fixed_matrices(self, device):
        # Assemble in CPU FP64, independently of the network's AMP/dtype policy.
        centres = (torch.arange(self.image_size, dtype=torch.float64) + .5) / self.image_size
        basis = cosine_basis(centres, self.image_size)
        derivative_basis = cosine_basis(centres, self.image_size, derivative=True)
        derivative = derivative_basis @ basis.T
        for name, matrix in (("derivative", derivative),):
            self.register_buffer(name, matrix.to(device=device, dtype=torch.float32),
                                 persistent=False)
            self.register_buffer(name + "_double", matrix.to(device), persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Module.float()/half()/double() also cast buffers. Reconstruct instead
        # of widening an already-rounded FP16/BF16 matrix back to FP32/FP64.
        if (self.derivative.dtype != torch.float32 or
                self.derivative_double.dtype != torch.float64):
            self._register_fixed_matrices(self.derivative.device)
        return self

    def _fixed_matrix(self, stem: str, value: torch.Tensor) -> torch.Tensor:
        if value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("cosine operators require float16/bfloat16/float32/float64")
        suffix = "_double" if value.dtype == torch.float64 else ""
        # Module/device migration moves every non-persistent buffer once.  This
        # call is then a no-op instead of allocating a cast matrix at every NFE.
        return getattr(self, stem + suffix).to(device=value.device)

    def spatial_derivatives(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (value.ndim != 4 or value.shape[-2:] != (self.image_size, self.image_size) or
                not value.is_floating_point()):
            raise ValueError("value must be floating [B,C,H,W] at the configured size")
        derivative = self._fixed_matrix("derivative", value)
        # Casting inputs alone does not escape an enclosing autocast context.
        with torch.autocast(device_type=value.device.type, enabled=False):
            value = value.to(derivative.dtype)
            dx = torch.matmul(value, derivative.T)
            dy = torch.matmul(derivative, value)
        return dx, dy
