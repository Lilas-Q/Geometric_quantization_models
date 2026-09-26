"""Single-channel SSIM for the LHFM-V evaluation protocol.

The uniform-window calculation is adapted from scikit-image v0.19.3
skimage/metrics/_structural_similarity.py.
Copyright (C) 2019, the scikit-image team. All rights reserved.
See licenses/scikit-image.txt for the BSD license.

This is deliberately not a general replacement for structural_similarity:
it implements the FP32, single-channel PredFormer evaluation convention.
"""
import numpy as np
from scipy.ndimage import uniform_filter


def _uniform_ssim(x, y):
    """Return one FP32 channel score for two FP32 spatial arrays."""
    ux = uniform_filter(x, size=7)
    uy = uniform_filter(y, size=7)
    uxx = uniform_filter(x * x, size=7)
    uyy = uniform_filter(y * y, size=7)
    uxy = uniform_filter(x * y, size=7)
    cov_norm = 49 / 48
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    c1, c2 = (.01 * 2)**2, (.03 * 2)**2
    a1, a2 = 2 * ux * uy + c1, 2 * vxy + c2
    b1, b2 = ux**2 + uy**2 + c1, vx + vy + c2
    score_map = (a1 * a2) / (b1 * b2)
    # The original multichannel wrapper stores each FP64 spatial mean in
    # an FP32 array before taking the channel mean, even for one channel.
    return np.float32(score_map[3:-3, 3:-3].mean(dtype=np.float64))


def frame_ssim(prediction, target):
    """FP64 [B,T] scores; predictions only are clipped to [0,1].

    Inputs have shape [B,T,1,H,W]. SSIM uses FP32 maps, a 7x7 uniform
    window, sample covariance, data_range=2, and a three-pixel crop.
    The PredFormer frame transform swapaxes(0,2) is retained exactly.
    """
    p = np.asarray(prediction, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32)
    if p.shape != y.shape or p.ndim != 5 or p.shape[2] != 1:
        raise ValueError("expected matching [B,T,1,H,W]")
    if min(p.shape[-2:]) < 7:
        raise ValueError("SSIM requires at least 7x7 images")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise FloatingPointError("nonfinite evaluation")
    if y.min() < 0 or y.max() > 1:
        raise ValueError("target must be [0,1]")
    p = np.minimum(np.maximum(p, 0), 1)
    scores = np.empty(p.shape[:2], dtype=np.float64)
    for b in range(p.shape[0]):
        for t in range(p.shape[1]):
            x = p[b, t].swapaxes(0, 2)[..., 0]
            z = y[b, t].swapaxes(0, 2)[..., 0]
            scores[b, t] = _uniform_ssim(x, z)
    return scores
