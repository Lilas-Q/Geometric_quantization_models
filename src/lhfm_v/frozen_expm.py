"""Signed shifted exponential for frozen second-order upwind transport.

A2 uses center -3/2, upstream +2 and second-upstream -1/2 per
positive speed component. Set q=1.5*max(1,max(|ux|+|uy|)), P=I+A2/q.
P is signed, not stochastic; ||P||_infinity <= 5/3. For mu=q*h<=3,
the order-24 tail is bounded by exp(-mu)*sum_{k>24}(5*mu/3)^k/k!
<1.2e-9 times (||J||inf+1.5||r||inf/q) per piece in exact arithmetic.
This scaling keeps the piece count equal to v7 for the same frozen velocity.
This truncation bound is not a positivity or long-rollout stability guarantee.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

TERMS = 24
MAX_SCALED_SHIFT = 3.0
SHIFT_MULTIPLIER = 1.5
MAX_INTERNAL_PIECES = 128


def upwind_generator(image, velocity):
    ux, uy = velocity[:, :1], velocity[:, 1:]
    left = F.pad(image[..., :-1], (1, 0, 0, 0))
    right = F.pad(image[..., 1:], (0, 1, 0, 0))
    above = F.pad(image[..., :-1, :], (0, 0, 1, 0))
    below = F.pad(image[..., 1:, :], (0, 0, 0, 1))
    left2 = F.pad(image[..., :-2], (2, 0, 0, 0))
    right2 = F.pad(image[..., 2:], (0, 2, 0, 0))
    above2 = F.pad(image[..., :-2, :], (0, 0, 2, 0))
    below2 = F.pad(image[..., 2:, :], (0, 0, 0, 2))
    return (ux.clamp_min(0) * (2*left-1.5*image-.5*left2)
            + (-ux).clamp_min(0) * (2*right-1.5*image-.5*right2)
            + uy.clamp_min(0) * (2*above-1.5*image-.5*above2)
            + (-uy).clamp_min(0) * (2*below-1.5*image-.5*below2))


def stencil_coefficients(source, velocity, q):
    ux, uy = velocity[:, :1], velocity[:, 1:]
    left, right = ux.clamp_min(0)/q, (-ux).clamp_min(0)/q
    above, below = uy.clamp_min(0)/q, (-uy).clamp_min(0)/q
    return torch.stack((1-1.5*(left+right+above+below),
        2*left, 2*right, 2*above, 2*below,
        -.5*left, -.5*right, -.5*above, -.5*below, source/q))


def exponential_stage(value, result, center_w, left_w, right_w, above_w, below_w,
                      left2_w, right2_w, above2_w, below2_w, forcing, weight):
    value = (center_w*value
             + left_w*F.pad(value[..., :-1], (1, 0, 0, 0))
             + right_w*F.pad(value[..., 1:], (0, 1, 0, 0))
             + above_w*F.pad(value[..., :-1, :], (0, 0, 1, 0))
             + below_w*F.pad(value[..., 1:, :], (0, 0, 0, 1))
             + left2_w*F.pad(value[..., :-2], (2, 0, 0, 0))
             + right2_w*F.pad(value[..., 2:], (0, 2, 0, 0))
             + above2_w*F.pad(value[..., :-2, :], (0, 0, 2, 0))
             + below2_w*F.pad(value[..., 2:, :], (0, 0, 0, 2))
             + forcing)
    return value, result + weight*value


def exponential_piece(image, source, velocity, q, poisson_mean, stage_fn=None):
    """Orders 0..24 of exp(-mu) exp(mu*P), including transported forcing.

    Every loop applies the affine map P*value+r/q. The source is therefore
    integrated through the same flow, not merely added as dt*r at the end.
    Both r/u stay attached; q is a detached numerical scaling choice only.
    """
    coeff = stencil_coefficients(source, velocity, q)
    value = image
    weight = torch.exp(-poisson_mean)
    result = weight*value
    stage_fn = exponential_stage if stage_fn is None else stage_fn
    for order in range(1, TERMS+1):
        weight = weight * poisson_mean / order
        value, result = stage_fn(value, result, *coeff.unbind(0), weight)
    return result


def frozen_exponential_step(image, source, velocity, dt, piece_fn=None):
    """Freeze once per neural interval; scaling introduces NO new field calls.

    The one host scalar chooses a bounded stable exponential approximation.
    It is deliberately outside compiled kernels. Detaching this numerical
    shift does not detach physical r/u or their contribution to the solution.
    """
    dtype = torch.float64 if image.dtype == torch.float64 else torch.float32
    image, source, velocity = image.to(dtype), source.to(dtype), velocity.to(dtype)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('positive finite frozen interval required')
    rate = float(velocity.detach().abs().sum(1).amax())
    if not math.isfinite(rate):
        raise FloatingPointError('nonfinite frozen transport field')
    rate = SHIFT_MULTIPLIER*max(1.0, rate)
    pieces = max(1, math.ceil(dt*rate/MAX_SCALED_SHIFT))
    if pieces > MAX_INTERNAL_PIECES:
        raise FloatingPointError('frozen transport rate exceeds numerical work budget; no velocity clipping')
    q = image.new_tensor(rate)
    mean = image.new_tensor(dt*rate/pieces)
    piece_fn = exponential_piece if piece_fn is None else piece_fn
    for _ in range(pieces):
        image = piece_fn(image, source, velocity, q, mean)
    return image
