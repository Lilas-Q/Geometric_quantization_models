"""Independent-pair linear conditional flow matching, noise t=0 -> data t=1."""

from dataclasses import dataclass
import torch
from .geometry import validate_images


def require(condition: torch.Tensor, message: str):
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class FlowBatch:
    image: torch.Tensor
    time: torch.Tensor
    target: torch.Tensor


def conditional_path(data, *, time=None, noise=None, generator=None):
    validate_images(data)
    if data.dtype not in (torch.float32, torch.float64):
        data = data.float()
    if time is None:
        time = torch.rand(data.shape[0], device=data.device, dtype=data.dtype, generator=generator)
    time = torch.as_tensor(time, device=data.device, dtype=data.dtype)
    if time.ndim == 0:
        time = time.expand(data.shape[0])
    if time.shape != (data.shape[0],):
        raise ValueError("time must be scalar or [B]")
    require(torch.isfinite(time).all() & (time >= 0).all() & (time <= 1).all(), "time outside [0,1]")
    if noise is None:
        noise = torch.randn(data.shape, device=data.device, dtype=data.dtype, generator=generator)
    if noise.shape != data.shape or noise.device != data.device or not noise.is_floating_point():
        raise ValueError("noise must match data")
    noise = noise.to(data.dtype)
    t = time[:, None, None, None]
    return FlowBatch((1 - t) * noise + t * data, time, data - noise)


def velocity_loss(prediction, batch: FlowBatch):
    if prediction.shape != batch.target.shape:
        raise ValueError("predicted velocity shape mismatch")
    # Pixel/coefficients L2, NOT gradient-weighted ambient momentum L2.
    dtype = torch.float64 if batch.target.dtype == torch.float64 else torch.float32
    return (prediction.to(dtype) - batch.target.to(dtype)).square().mean()
