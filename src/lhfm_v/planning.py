"""Joint temporal translation inspired by SimVP/gSTA, independently implemented.

Time is folded into channels before translation. Normalization is per example
and contains no batch statistics. Gates act on features, never on physical u.
"""
import math
import torch
from torch import nn


def norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


class TemporalBlock(nn.Module):
    def __init__(self, channels, expansion):
        super().__init__()
        c = channels
        self.norm1 = norm(c)
        self.project_in = nn.Conv2d(c, c, 1)
        self.activation = nn.GELU()
        self.local = nn.Conv2d(c, c, 5, padding=2, groups=c)
        self.dilated = nn.Conv2d(c, c, 7, padding=9, dilation=3, groups=c)
        self.content_gate = nn.Conv2d(c, 2*c, 1)
        self.project_out = nn.Conv2d(c, c, 1)
        self.norm2 = norm(c)
        self.ffn = nn.Sequential(
            nn.Conv2d(c, expansion*c, 1),
            nn.Conv2d(expansion*c, expansion*c, 3, padding=1, groups=expansion*c),
            nn.GELU(), nn.Conv2d(expansion*c, c, 1))
        self.attention_scale = nn.Parameter(torch.full((1, c, 1, 1), .01))
        self.ffn_scale = nn.Parameter(torch.full((1, c, 1, 1), .01))

    def forward(self, x):
        feature = self.activation(self.project_in(self.norm1(x)))
        content, gate = self.content_gate(self.dilated(self.local(feature))).chunk(2, 1)
        x = x + self.attention_scale * self.project_out(content * gate.sigmoid())
        return x + self.ffn_scale * self.ffn(self.norm2(x))


class FuturePlanner(nn.Module):
    def __init__(self,latent_channels,channels,depth,expansion):
        super().__init__()
        self.channels=channels
        self.input=nn.Conv2d(latent_channels,channels,1)
        self.blocks=nn.Sequential(*(TemporalBlock(channels,expansion) for _ in range(depth)))
        self.norm=norm(channels)
        # One joint channel projection, then split into ten ordered future plans.
        self.output=nn.Linear(channels,10*channels)

    def forward(self,z0):
        x=self.input(z0.permute(0,3,1,2))
        x=self.norm(self.blocks(x)).permute(0,2,3,1)
        b,h,w,_=x.shape
        return self.output(x).reshape(b,h,w,10,self.channels).permute(0,3,1,2,4).contiguous()


class StateCorrection(nn.Module):
    def __init__(self,latent_channels,plan_channels,hidden):
        super().__init__()
        self.features=nn.Sequential(nn.LayerNorm(latent_channels+plan_channels),
            nn.Linear(latent_channels+plan_channels,hidden),nn.SiLU())
        # NHWC Linear readouts preserve eager/AOT gradients in both precisions.
        self.source=nn.Linear(hidden,16)
        self.transport=nn.Linear(hidden,32)
        for head in (self.source,self.transport):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def forward(self,z,plan):
        feature=self.features(torch.cat((z,plan),-1))
        return torch.cat((self.source(feature),self.transport(feature)),-1)
