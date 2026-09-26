"""Encode ordered history once; large residual blocks operate at H/4 x W/4."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def norm(channels):
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1, self.norm2 = norm(channels), norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        return x + self.conv2(F.silu(self.norm2(h)))


class ChannelMLP(nn.Module):
    def __init__(self, channels, expansion):
        super().__init__()
        self.norm = norm(channels)
        self.expand = nn.Conv2d(channels, channels*expansion, 1)
        self.project = nn.Conv2d(channels*expansion, channels, 1)

    def forward(self, x):
        return x + self.project(F.silu(self.expand(self.norm(x))))


class HistoryLatentEncoder(nn.Module):
    def __init__(self, context_frames, widths, latent_channels, depth, expansion, channels_last, spatial_groups):
        super().__init__()
        self.channels_last = channels_last
        self.latent_channels = latent_channels
        self.spatial_groups = spatial_groups
        a, b, c = widths
        self.stem = nn.Conv2d(2*context_frames+1, a, 1)
        self.down1 = nn.Conv2d(a, b, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(b, c, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(*[Residual(c) for _ in range(depth)],
                                        ChannelMLP(c, expansion))
        self.output_norm = norm(c)
        # z0, omega/decay/feedback/drift, group-specific five-point spatial weights, and
        # 3 fields x 16 pixel phases. No full-resolution learned basis cache.
        self.output = nn.Conv2d(c, 4*latent_channels+5*spatial_groups+48, 1)
        nn.init.normal_(self.output.weight, std=.005)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            d = latent_channels
            nn.init.normal_(self.output.weight[:d], std=.02)
            self.output.bias[d:d+d//2].copy_(torch.linspace(.03, .5, d//2))
            self.output.bias[d+d//2:2*d].fill_(-4.)
            self.output.bias[4*d:4*d+5*spatial_groups:5].fill_(3.)
            nn.init.normal_(self.output.weight[-48:], std=1e-3)

    def forward(self, context):
        batch, _, _, height, width = context.shape
        images = context[:, :, 0].float()
        x = (torch.arange(width, device=context.device, dtype=torch.float32)+.5)*(2./width)-1.
        y = (torch.arange(height, device=context.device, dtype=torch.float32)+.5)*(2./height)-1.
        coords = torch.cat((x.reshape(1,1,1,width).expand(batch,1,height,width),
                            y.reshape(1,1,height,1).expand(batch,1,height,width)), 1)
        value = torch.cat((images, images[:,1:]-images[:,:-1], coords), 1)
        if self.channels_last:
            value = value.contiguous(memory_format=torch.channels_last)
        value = F.leaky_relu(self.stem(value), negative_slope=.2)
        value = F.leaky_relu(self.down1(value), negative_slope=.2)
        value = F.leaky_relu(self.down2(value), negative_slope=.2)
        value = self.output(F.silu(self.output_norm(self.bottleneck(value))))
        value = value.permute(0,2,3,1).contiguous()
        d = self.latent_channels
        return value[...,:d], value[...,d:4*d], value[...,4*d:4*d+5*self.spatial_groups].reshape(batch,height//4,width//4,self.spatial_groups,5), value[...,-48:]
