"""Budgeted image resizing and a compact dynamic-resolution vision transformer.

Spatial RoPE is applied to Q/K. This is an independent compact implementation,
not a checkpoint-compatible copy of Qwen2.5-VL or its language-side MRoPE.
"""

import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SmartResizeConfig:
    patch_size: int = 14
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    factor: int = 28

    def smart_resize(self, height, width):
        f = self.factor
        if (
            min(height, width, f) <= 0
            or self.min_pixels <= 0
            or self.max_pixels < self.min_pixels
        ):
            raise ValueError("Invalid size or pixel budget")
        if max(height, width) / min(height, width) > 200:
            raise ValueError("Aspect ratio exceeds 200")
        h = max(f, round(height / f) * f)
        w = max(f, round(width / f) * f)
        if h * w > self.max_pixels:
            s = math.sqrt(self.max_pixels / (height * width))
            h = max(f, math.floor(height * s / f) * f)
            w = max(f, math.floor(width * s / f) * f)
        elif h * w < self.min_pixels:
            s = math.sqrt(self.min_pixels / (height * width))
            h = math.ceil(height * s / f) * f
            w = math.ceil(width * s / f) * f
        if not self.min_pixels <= h * w <= self.max_pixels:
            raise ValueError(
                "Budget cannot satisfy rounded dimensions; widen budget or reduce aspect ratio"
            )
        return h, w


class MRotaryEmbedding2D(nn.Module):
    """Separate row/column rotations, using interleaved pairs in each axis."""

    def __init__(self, dim, base=10000.0):
        super().__init__()
        if dim % 4:
            raise ValueError("head dimension must be divisible by 4")
        self.dim = dim
        self.register_buffer(
            "inv_freq",
            base ** (-torch.arange(dim // 4).float() / (dim // 4)),
            persistent=False,
        )

    def forward(self, x, grid_h, grid_w):
        # x: B,N,D or B,N,heads,D
        if x.shape[1] != grid_h * grid_w or x.shape[-1] != self.dim:
            raise ValueError("Invalid rotary input")
        row = torch.arange(grid_h, device=x.device).repeat_interleave(grid_w)
        col = torch.arange(grid_w, device=x.device).repeat(grid_h)
        result = []
        for part, pos in zip(x.chunk(2, dim=-1), (row, col)):
            angle = pos[:, None] * self.inv_freq[None, :]
            angle = angle.reshape(1, len(pos), *([1] * (x.ndim - 3)), -1)
            a, b = part[..., 0::2], part[..., 1::2]
            cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
            result.append(
                torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
            )
        return torch.cat(result, dim=-1)


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, full_attn_layers=(7, 14, 21, 28)):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must divide heads")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.full_attn_layers = set(full_attn_layers)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rope = MRotaryEmbedding2D(dim // num_heads)

    def forward(self, x, grid_h, grid_w, layer_idx=0):
        B, N, C = x.shape
        h = self.num_heads
        d = C // h
        q, k, v = self.qkv(x).reshape(B, N, 3, h, d).unbind(2)
        q = self.rope(q, grid_h, grid_w)
        k = self.rope(k, grid_h, grid_w)
        if layer_idx in self.full_attn_layers:
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
            return self.proj(y.transpose(1, 2).reshape(B, N, C))
        ws = self.window_size
        ph = (-grid_h) % ws
        pw = (-grid_w) % ws
        gh, gw = grid_h + ph, grid_w + pw

        def windows(t):
            t = t.reshape(B, grid_h, grid_w, C).permute(0, 3, 1, 2)
            t = F.pad(t, (0, pw, 0, ph)).permute(0, 2, 3, 1)
            return (
                t.reshape(B, gh // ws, ws, gw // ws, ws, h, d)
                .permute(0, 1, 3, 5, 2, 4, 6)
                .reshape(-1, h, ws * ws, d)
            )

        valid = torch.ones(1, grid_h, grid_w, device=x.device, dtype=torch.bool)
        valid = F.pad(valid, (0, pw, 0, ph), value=False)
        valid = (
            valid.reshape(1, gh // ws, ws, gw // ws, ws)
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, ws * ws)
            .repeat(B, 1)
        )
        y = F.scaled_dot_product_attention(
            windows(q), windows(k), windows(v), attn_mask=valid[:, None, None, :]
        )
        y = (
            y.reshape(B, gh // ws, gw // ws, h, ws, ws, d)
            .permute(0, 1, 4, 2, 5, 3, 6)
            .reshape(B, gh, gw, C)
        )
        return self.proj(y[:, :grid_h, :grid_w].reshape(B, N, C))


class QwenViTBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=8,
        mlp_ratio=4.0,
        full_attn_layers=(7, 14, 21, 28),
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size, full_attn_layers)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, h, w, layer_idx):
        x = x + self.attn(self.norm1(x), h, w, layer_idx)
        return x + self.mlp(self.norm2(x))


class QwenViT(nn.Module):
    """Compact research ViT; name retained for compatibility with early examples."""

    def __init__(
        self,
        img_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=768,
        depth=8,
        num_heads=12,
        window_size=8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.blocks = nn.ModuleList(
            [
                QwenViTBlock(
                    embed_dim, num_heads, window_size, full_attn_layers={depth - 1}
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if x.shape[-2] % self.patch_size or x.shape[-1] % self.patch_size:
            raise ValueError("Resize image to patch multiples first")
        x = self.patch_embed(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        for i, b in enumerate(self.blocks):
            x = b(x, h, w, i)
        return self.norm(x).mean(1)
