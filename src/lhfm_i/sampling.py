"""ODE in image coordinates, representing the whole evolving exact graph.

No independent point clouds, score conversion, injected Brownian noise or
intermediate clamping/projection. Euler/Heun coefficient updates stay inside
the exact graph family; this does not assert a common ambient symplectic map
for the nonlinear image-to-image integrator.
"""

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class Sample:
    images: torch.Tensor
    nfe: int
    maximum_rms: float


@torch.no_grad()
def sample_ode(model, shape, *, steps=128, method="heun", device="cpu", generator=None,
               initial=None, bfloat16=False, maximum_rms=100.):
    device = torch.device(device)
    if (isinstance(steps, bool) or not isinstance(steps, int) or steps < 1 or
            method not in {"euler", "heun"} or len(shape) != 4 or shape[1] != 3 or
            any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in shape) or
            not math.isfinite(maximum_rms) or maximum_rms <= 0):
        raise ValueError("invalid sampling options")
    if bfloat16 and device.type != "cuda":
        raise ValueError("BF16 sampler is CUDA-only")
    if initial is not None and (tuple(initial.shape) != tuple(shape) or not initial.is_floating_point()):
        raise ValueError("initial state shape/type mismatch")
    x = (torch.randn(shape, device=device, dtype=torch.float32, generator=generator) if initial is None
         else initial.to(device=device, dtype=torch.float32).clone())
    was_training = model.training
    model.eval()
    peak = torch.zeros((), device=device)
    nfe = 0
    def check(value):
        nonlocal peak
        # NaN/Inf in any component propagates into this RMS and then into peak.
        # This also rejects square overflow, as did the original guard.
        rms = value.square().flatten(1).mean(1).sqrt().amax()
        peak = torch.maximum(peak, rms)
    def velocity(value, time):
        nonlocal nfe
        output = model(value, torch.full((shape[0],), time, device=device, dtype=torch.float32))
        if output.shape != value.shape:
            raise ValueError("velocity shape mismatch")
        nfe += 1
        return output.float()
    try:
        check(x)
        dt = 1 / steps
        # One autocast scope keeps immutable eval-weight casts cached across NFEs.
        # States and all Euler/Heun arithmetic remain FP32; only model ops autocast.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bfloat16):
            for k in range(steps):
                v = velocity(x, k / steps)
                predictor = x + dt * v
                check(predictor)
                if method == "heun":
                    x = x + (.5 * dt) * (v + velocity(predictor, (k + 1) / steps))
                    check(x)
                else:
                    x = predictor
        # Only the scalar peak needs transfer, once per complete sampling batch.
        maximum = float(peak)
        if not math.isfinite(maximum) or maximum > maximum_rms:
            raise FloatingPointError("ODE nonfinite state or RMS bound violation; no clipping applied")
        return Sample(x, nfe, maximum)
    finally:
        model.train(was_training)
