"""Original physical-rollout objective plus direct coarse future-plan supervision."""
import math
import torch
from torch.nn import functional as F


def coarse_future_target(target):
    """Pool each real frame independently; no positions, masks or future encoder."""
    if target.ndim != 5 or target.shape[2] != 1 or any(s % 4 for s in target.shape[-2:]):
        raise ValueError("future targets must be [B,T,1,H,W] with spatial dimensions divisible by four")
    b,t,c,h,w=target.shape
    return F.avg_pool2d(target.detach().float().reshape(b*t,c,h,w),4,4).reshape(b,t,c,h//4,w//4)


def forecast_loss(forecast, target, *, source_weight=1e-3, velocity_tv_weight=1e-4,
                  planning_weight=0.05):
    if target.shape != forecast.prediction.shape:
        raise ValueError("future targets must match the entire predicted sequence")
    if not math.isfinite(planning_weight) or planning_weight < 0:
        raise ValueError("planning weight must be finite and nonnegative")
    mse = (forecast.prediction.float() - target.float()).square().mean()
    total = mse + source_weight * forecast.source_energy + velocity_tv_weight * forecast.velocity_tv
    planning_mse=mse.new_zeros(())
    if planning_weight > 0:
        if forecast.planned_coarse is None:
            raise ValueError("positive planning weight requires supervise_plan=True during training")
        pooled=coarse_future_target(target)
        if forecast.planned_coarse.shape != pooled.shape:
            raise ValueError("coarse plan must match every pooled future frame")
        planning_mse=(forecast.planned_coarse.float()-pooled).square().mean()
        total=total+planning_weight*planning_mse
    return total, {"loss": total.detach(), "pixel_mse_raw": mse.detach(),
                   "planning_mse_raw": planning_mse.detach(),
                   "planning_loss_weighted": (planning_weight*planning_mse).detach(),
                   "source_energy": forecast.source_energy.detach(),
                   "velocity_tv": forecast.velocity_tv.detach(),
                   "r_rms": forecast.r_rms.detach(), "u_rms": forecast.u_rms.detach()}
