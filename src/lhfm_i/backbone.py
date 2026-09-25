"""Shared residual/attention U-Net backbone used by LHFM-I."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def groups(channels):
    return next(g for g in range(min(32, channels), 0, -1) if channels % g == 0)


def zero(module):
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class TimeEmbedding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.register_buffer("frequencies", torch.exp(-math.log(10000) *
            torch.arange(channels // 2) / max(channels // 2 - 1, 1)), persistent=False)
        self.layers = nn.Sequential(nn.Linear(channels, channels * 4), nn.SiLU(),
                                    nn.Linear(channels * 4, channels * 4))

    def forward(self, time):
        phase = 1000 * time[:, None].float() * self.frequencies[None]
        return self.layers(torch.cat((phase.cos(), phase.sin()), -1))


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, time_channels, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups(incoming), incoming, eps=1e-6)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.time = nn.Linear(time_channels, 2 * outgoing)
        self.norm2 = nn.GroupNorm(groups(outgoing), outgoing, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero(nn.Conv2d(outgoing, outgoing, 3, padding=1))
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1)

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(F.silu(t)).chunk(2, 1)
        h = self.norm2(h) * (1 + scale[..., None, None]) + shift[..., None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return (self.skip(x) + h) * (2 ** -.5)


class Attention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(groups(channels), channels, eps=1e-6)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.out = zero(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        def arrange(value):
            return value.reshape(b, self.heads, c // self.heads, h * w).transpose(-1, -2)
        hidden = F.scaled_dot_product_attention(arrange(q), arrange(k), arrange(v), dropout_p=0.)
        hidden = hidden.transpose(-1, -2).reshape(b, c, h, w)
        return (x + self.out(hidden)) * (2 ** -.5)


class Block(nn.Module):
    def __init__(self, incoming, outgoing, time_channels, dropout, attention, heads):
        super().__init__()
        self.residual = ResidualBlock(incoming, outgoing, time_channels, dropout)
        self.attention = Attention(outgoing, heads) if attention else nn.Identity()

    def forward(self, x, t):
        return self.attention(self.residual(x, t))


class ImageVelocityUNet(nn.Module):
    def __init__(self, *, image_size=32, base_channels=128, multipliers=(1, 2, 2, 2),
                 residual_blocks=2, attention_resolutions=(16, 8), heads=4, dropout=.1):
        super().__init__()
        if (base_channels < 4 or base_channels % 2 or min(multipliers) < 1 or
                image_size % 2 ** (len(multipliers) - 1) or heads < 1 or
                any(base_channels * m % heads for m in multipliers) or
                residual_blocks < 1 or not 0 <= dropout < 1):
            raise ValueError("invalid U-Net dimensions or dropout")
        self.image_size = image_size
        self.time = TimeEmbedding(base_channels)
        self.input = nn.Conv2d(3, base_channels, 3, padding=1)
        self.down = nn.ModuleList()
        channels, resolution = base_channels, image_size
        skips = [channels]
        for level, multiplier in enumerate(multipliers):
            blocks = nn.ModuleList()
            for _ in range(residual_blocks):
                outgoing = base_channels * multiplier
                blocks.append(Block(channels, outgoing, base_channels * 4, dropout,
                                    resolution in attention_resolutions, heads))
                channels = outgoing
                skips.append(channels)
            downsample = nn.Conv2d(channels, channels, 3, stride=2, padding=1) if level < len(multipliers) - 1 else nn.Identity()
            self.down.append(nn.ModuleDict(dict(blocks=blocks, sample=downsample)))
            if level < len(multipliers) - 1:
                skips.append(channels)
                resolution //= 2
        self.middle1 = Block(channels, channels, base_channels * 4, dropout, True, heads)
        self.middle2 = ResidualBlock(channels, channels, base_channels * 4, dropout)
        self.up = nn.ModuleList()
        for level in reversed(range(len(multipliers))):
            blocks = nn.ModuleList()
            for _ in range(residual_blocks + 1):
                outgoing = base_channels * multipliers[level]
                blocks.append(Block(channels + skips.pop(), outgoing, base_channels * 4,
                                    dropout, resolution in attention_resolutions, heads))
                channels = outgoing
            sample = nn.Conv2d(channels, channels, 3, padding=1) if level > 0 else nn.Identity()
            self.up.append(nn.ModuleDict(dict(blocks=blocks, sample=sample)))
            if level > 0:
                resolution *= 2
        assert not skips
        self.norm = nn.GroupNorm(groups(channels), channels, eps=1e-6)
        self.output = zero(nn.Conv2d(channels, 3, 3, padding=1))

    def forward(self, image, time):
        if image.ndim != 4 or image.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError("expected RGB image with configured size")
        if time.shape != (image.shape[0],):
            raise ValueError("expected time [B]; no class conditioning is accepted")
        t = self.time(time)
        h = self.input(image)
        skips = [h]
        for level, stage in enumerate(self.down):
            for block in stage["blocks"]:
                h = block(h, t)
                skips.append(h)
            if level < len(self.down) - 1:
                h = stage["sample"](h)
                skips.append(h)
        h = self.middle2(self.middle1(h, t), t)
        for level, stage in enumerate(self.up):
            for block in stage["blocks"]:
                h = block(torch.cat((h, skips.pop()), 1), t)
            if level < len(self.up) - 1:
                h = stage["sample"](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.output(F.silu(self.norm(h)))
