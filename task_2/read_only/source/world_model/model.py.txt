"""Small conditional U-Net predicting a pixel-space velocity field."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, channels)
        self.norm2 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, condition):
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.condition(condition)[:, :, None, None]
        return x + self.conv2(F.silu(self.norm2(hidden)))


class WorldModel(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(16, width * 2), nn.SiLU(), nn.Linear(width * 2, width * 2))
        self.action_embedding = nn.Embedding(3, width * 2)
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.enc = ResidualBlock(width, width * 2)
        self.down = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.mid1 = ResidualBlock(width * 2, width * 2)
        self.mid2 = ResidualBlock(width * 2, width * 2)
        self.up = nn.Conv2d(width * 2, width, 3, padding=1)
        self.dec = ResidualBlock(width, width * 2)
        self.out = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_t: torch.Tensor, time: torch.Tensor, context: torch.Tensor, action: torch.Tensor):
        """Inputs: x_t [B,1,H,W], time [B], context [B,2,1,H,W], action [B]."""
        frequencies = 2 ** torch.arange(8, device=x_t.device, dtype=x_t.dtype)
        phases = time[:, None] * frequencies[None, :] * math.pi
        time_condition = self.time_mlp(torch.cat([phases.sin(), phases.cos()], dim=-1))
        action_condition = self.action_embedding(action).mean(dim=0, keepdim=True)
        condition = time_condition + action_condition
        skip = self.enc(self.stem(torch.cat([x_t, context.flatten(1, 2)], dim=1)), condition)
        hidden = self.mid2(self.mid1(self.down(skip), condition), condition)
        hidden = self.up(F.interpolate(hidden, size=skip.shape[-2:], mode="nearest"))
        return self.out(self.dec(hidden + skip, condition))
