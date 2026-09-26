"""Deterministic 10-to-10 metrics; report raw and clipped predictions separately."""
import torch
from torch.nn import functional as F


PROTOCOL = dict(scale="uint8/255", conditioning="first10; autoregressive future10",
    stochastic_samples=1, prediction="deterministic; no noise or best-of-N",
    primary="MSE raw; mean pixels/channels, then equally frames/sequences",
    spatial_sum="pixel_mean * H * W; other protocol differences still require alignment",
    ssim="clipped [0,1]; Gaussian 11x11 sigma1.5, valid, population, K1=.01 K2=.03",
    psnr="clipped; per-frame -10log10(max(MSE,1e-12)), then average; cap120dB")


@torch.no_grad()
def metric_tensors(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[2] != 1:
        raise ValueError("expected matching [B,T,1,H,W]")
    p, y = prediction.double(), target.double()
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise FloatingPointError("nonfinite evaluation")
    if y.min() < 0 or y.max() > 1:
        raise ValueError("target must be [0,1]")
    b, t, _, h, w = p.shape
    raw_error = p - y
    clipped = p.clamp(0, 1)
    error = clipped - y
    mse = error.square().mean((2, 3, 4))
    raw_mse = raw_error.square().mean((2, 3, 4))
    if min(h, w) < 11:
        raise ValueError("SSIM requires at least 11x11 images")
    a = torch.arange(11, dtype=p.dtype, device=p.device) - 5
    g = (-a.square() / (2 * 1.5**2)).exp()
    g /= g.sum()
    kernel = (g[:, None] * g[None, :])[None, None]
    x, z = clipped.reshape(-1, 1, h, w), y.reshape(-1, 1, h, w)
    ux, uy = F.conv2d(x, kernel), F.conv2d(z, kernel)
    vx, vy = F.conv2d(x*x, kernel) - ux*ux, F.conv2d(z*z, kernel) - uy*uy
    cov = F.conv2d(x*z, kernel) - ux*uy
    ssim = ((2*ux*uy + .01**2) * (2*cov + .03**2)
            / ((ux*ux + uy*uy + .01**2) * (vx + vy + .03**2))).mean((1, 2, 3)).reshape(b, t)
    return dict(mse_raw=raw_mse, mse_spatial_sum_raw=raw_mse*h*w,
        mae_raw=raw_error.abs().mean((2, 3, 4)),
        mae_spatial_sum_raw=raw_error.abs().mean((2, 3, 4))*h*w,
        mse_clipped=mse, mae_clipped=error.abs().mean((2, 3, 4)),
        ssim_clipped=ssim, psnr_clipped=-10*torch.log10(mse.clamp_min(1e-12)),
        out_of_range_fraction=((p < 0) | (p > 1)).double().mean((2, 3, 4)))
