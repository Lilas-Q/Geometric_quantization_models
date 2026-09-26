"""Deterministic 10-to-10 metrics; report raw and clipped predictions separately."""
import torch

from .ssim import frame_ssim


PROTOCOL = dict(scale="uint8/255", conditioning="first10; autoregressive future10",
    stochastic_samples=1, prediction="deterministic; no noise or best-of-N",
    primary="MSE raw; mean pixels/channels, then equally frames/sequences",
    spatial_sum="pixel_mean * H * W; other protocol differences still require alignment",
    ssim="prediction clipped [0,1]; float32; scikit-image 0.19.3 protocol; "
         "uniform 7x7; sample covariance; data_range=2; K1=.01 K2=.03; "
         "swapaxes(0,2); crop3; float64 spatial mean, float32 channel mean; "
         "equal frames/sequences in float64",
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
    # Match the published evaluation's FP32 arrays and per-frame reduction.
    # SSIM runs on CPU; all other metrics retain their FP64 tensor reductions.
    scores = frame_ssim(prediction.float().cpu().numpy(),
                        target.float().cpu().numpy())
    ssim = torch.as_tensor(scores, dtype=torch.float64, device=p.device)
    return dict(mse_raw=raw_mse, mse_spatial_sum_raw=raw_mse*h*w,
        mae_raw=raw_error.abs().mean((2, 3, 4)),
        mae_spatial_sum_raw=raw_error.abs().mean((2, 3, 4))*h*w,
        mse_clipped=mse, mae_clipped=error.abs().mean((2, 3, 4)),
        ssim_clipped=ssim, psnr_clipped=-10*torch.log10(mse.clamp_min(1e-12)),
        out_of_range_fraction=((p < 0) | (p > 1)).double().mean((2, 3, 4)))
