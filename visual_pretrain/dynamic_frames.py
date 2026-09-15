"""Frame-budget scheduling, temporal sampling and masked within-video attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1) VideoBinPacker（动态帧 binpack）
# =====================================================================
@dataclass
class VideoBinPackerConfig:
    """视频帧 binpack 配置。

    笔记原文：
      "视频最大60帧，会设置最大浪费60，60÷1200=5%"
      "以 60 为例，控制 Max Token 的最大浪费比率，若间隙大于 60，任何视频都可塞入，最大浪费帧为 60"
    """

    bin_size: int = 1200  # 一个 bin 内的总帧数上限
    max_frames_per_video: int = 60  # 单个视频最大帧数（业务上限）
    max_waste_ratio: float = 0.05  # 5%

    @property
    def max_waste(self) -> int:
        # 60 / 1200 = 5%
        return int(self.bin_size * self.max_waste_ratio)


@dataclass
class VideoSample:
    video_id: str
    frame_paths: List[str]  # 每帧的路径
    num_frames: int  # 抽帧后的帧数（1 <= n <= 60）
    # 注意：抽帧后每帧本身还要做 smart resize 得到对应 token 数，但这里简化为"1 帧 = 1 token 单位"


def video_binpack(
    samples: List[VideoSample], cfg: VideoBinPackerConfig
) -> List[List[VideoSample]]:
    """贪心视频帧 binpack。

    算法（first-fit-decreasing 简化版）：
      - 按 num_frames 从大到小排序
      - 依次尝试塞进已有 bin，若所有 bin 都装不下（剩余 < max_waste），则开新 bin
      - 视频帧总数 <= bin_size 的 bin 视为有效

    Returns:
        List[bins]，每个 bin 是一个 VideoSample 列表。
        单 bin 内所有视频的 num_frames 之和 <= bin_size。
    """
    if cfg.bin_size <= 0 or cfg.max_frames_per_video <= 0:
        raise ValueError("Invalid video budget")
    if any(
        s.num_frames <= 0
        or s.num_frames > len(s.frame_paths)
        or min(s.num_frames, cfg.max_frames_per_video) > cfg.bin_size
        for s in samples
    ):
        raise ValueError("Video frames must fit bin budget and available paths")
    # 1) 截断到 max_frames_per_video
    samples = [
        VideoSample(
            s.video_id,
            s.frame_paths[: min(s.num_frames, cfg.max_frames_per_video)],
            min(s.num_frames, cfg.max_frames_per_video),
        )
        for s in samples
    ]
    # 2) 按 num_frames 降序
    samples = sorted(samples, key=lambda s: -s.num_frames)

    bins: List[List[VideoSample]] = []
    bin_used: List[int] = []

    for sample in samples:
        # 找剩余空间 >= sample.num_frames 的 bin
        placed = False
        for i, used in enumerate(bin_used):
            if used + sample.num_frames <= cfg.bin_size:
                bins[i].append(sample)
                bin_used[i] += sample.num_frames
                placed = True
                break
        if not placed:
            # 开新 bin
            bins.append([sample])
            bin_used.append(sample.num_frames)

    # 3) 过滤掉浪费率超阈值的 bin（demo 阶段不过滤太小的浪费；生产按业务调）
    filtered = []
    for i, used in enumerate(bin_used):
        filtered.append(bins[i])  # demo: 不过滤；生产可加 waste_ratio 阈值
    return filtered


# =====================================================================
# 2) FrameAttention（先 frame-window 再 frame-full，带分割线 mask）
# =====================================================================
class FrameAttention(nn.Module):
    """笔记原文：

      "视频表征基于之前动态分辨率方案，先在图片内部做表征（window attention 和 full attention）
       再对帧和帧之间做 window attention，最后对所有图片之间做 full attention"

      "为避免不同帧和视频之间信息关联，在帧和帧、视频和视频之间设置分割线"

    这里实现"帧间 attention"部分（图片内部表征由 dynamic_resolution.QwenViT 处理）。
    分割线 mask：用 segment_ids 控制，只有同一 segment_id 的 token 之间能 attend。
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    @staticmethod
    def _make_segment_mask(seg_ids: torch.Tensor) -> torch.Tensor:
        """生成 segment mask：(N, N)，同一 segment 才能 attend。

        seg_ids: (B, N) —— 每个 token 属于哪个 segment（视频或帧）
        Returns: (B, N, N) bool mask，True 表示允许 attend。
        """
        return seg_ids.unsqueeze(2) == seg_ids.unsqueeze(1)  # (B, N, N)

    def forward(
        self,
        x: torch.Tensor,  # (B, N, dim) — 多帧图片 embedding 拼起来
        seg_ids: torch.Tensor,  # (B, N)   — 每个 token 属于哪个视频
        use_full: bool = False,  # True: full attention；False: window attention
        window_size: int | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)

        # segment mask：禁止跨视频/跨帧 attention
        seg_mask = self._make_segment_mask(seg_ids).unsqueeze(1)  # (B, 1, N, N)

        if use_full or N <= (window_size or self.window_size):
            attn_mask = seg_mask  # SDPA: True allows attention
        else:
            # window attention：在 window 内 + 同一 segment
            ws = window_size or self.window_size
            idx = torch.arange(N, device=x.device)
            win_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() < ws  # (N, N)
            attn_mask = win_mask.unsqueeze(0).unsqueeze(0) & seg_mask

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class VideoEncoder(nn.Module):
    """视频编码器：frame-local window attn -> frame-full attn -> 平均池化 -> 1024 维向量。

    输入：每个视频的多帧图片已通过 QwenViT 表征成 (num_frames, embed_dim) 向量。
    输出：每个视频一个 (embed_dim,) 向量。
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, window_size: int = 4):
        super().__init__()
        self.frame_local = FrameAttention(embed_dim, num_heads, window_size=window_size)
        self.frame_full = FrameAttention(embed_dim, num_heads, window_size=window_size)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        video_frame_embeds: torch.Tensor,  # (B, T, embed_dim) —— B 个视频、T 帧
        frames_per_video: torch.Tensor,  # (B,) —— 每个视频实际帧数（用于 segment_id）
    ) -> torch.Tensor:
        """Returns: (B, embed_dim) 视频 embedding。"""
        B, T, C = video_frame_embeds.shape

        # 构造 segment_ids：第 i 个 token 属于视频 i // frames_per_video[i]
        # 但因为每个视频帧数不同，最简单：直接给每帧一个 video_id
        seg_ids = torch.zeros(B, T, dtype=torch.long, device=video_frame_embeds.device)
        start = 0
        for i, n in enumerate(frames_per_video.tolist()):
            seg_ids[i, start : start + n] = i
            start = 0  # T 是 padding 后每视频最大帧数，所以下个视频也从头开始
            # 但实际上我们 batch 里每个视频的 T 是相同的；这里简化处理

        # 实际更简单：每个视频 segment_id 就是自己的 i
        seg_ids = (
            torch.arange(B, device=video_frame_embeds.device)
            .unsqueeze(1)
            .expand(B, T)
            .contiguous()
        )
        # 屏蔽 padding 帧：用一个 special "padding" segment
        # 这里假设 frame_embeds[:, :frames_per_video[i]] 是真实帧，后面 padding
        valid_mask = torch.arange(T, device=video_frame_embeds.device).unsqueeze(
            0
        ) < frames_per_video.unsqueeze(1)  # (B, T)
        # padding 帧的 segment_id 设成 B（一个特殊的"padding"segment，attention 不到）
        seg_ids = torch.where(valid_mask, seg_ids, torch.full_like(seg_ids, B))

        # Frame-local attention
        h = video_frame_embeds + self.frame_local(
            self.norm1(video_frame_embeds), seg_ids, use_full=False
        )
        # Frame-full attention
        h = h + self.frame_full(self.norm2(h), seg_ids, use_full=True)

        # 平均池化（笔记："通过平均池化将多个向量表征为一个1024维向量"）
        # 只对有效帧做平均
        mask = valid_mask.float().unsqueeze(-1)  # (B, T, 1)
        video_emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, C)
        return video_emb


# =====================================================================
# Smoke test
# =====================================================================
if __name__ == "__main__":
    # 1) VideoBinPacker
    cfg = VideoBinPackerConfig(bin_size=1200, max_frames_per_video=60)
    samples = [
        VideoSample(f"v{i}", [f"frame_{j}.jpg" for j in range(n)], n)
        for i, n in enumerate([45, 30, 60, 15, 50, 20, 10, 55, 40, 25])
    ]
    bins = video_binpack(samples, cfg)
    print(f"[VideoBinPack] {len(samples)} videos -> {len(bins)} bins")
    for i, b in enumerate(bins):
        total_frames = sum(s.num_frames for s in b)
        print(
            f"  bin {i}: {len(b)} videos, total frames = {total_frames}/{cfg.bin_size}"
        )

    # 2) VideoEncoder forward
    enc = VideoEncoder(embed_dim=384, num_heads=6, window_size=4)
    # 假设每个视频 8 帧
    frame_emb = torch.randn(2, 8, 384)
    frames_per_video = torch.tensor([8, 8])
    video_emb = enc(frame_emb, frames_per_video)
    print(f"[VideoEncoder] frame_emb {frame_emb.shape} -> video_emb {video_emb.shape}")


def sample_frames(frames, token_budget, tokens_per_frame, min_frames=1):
    """Choose real frame indices using temporal coverage plus image-change scores.

    Input T,C,H,W. This deterministic heuristic is not a learned risk detector;
    it cannot guarantee recall of a short event. Budget is counted after resizing.
    """
    if frames.ndim != 4 or len(frames) == 0 or tokens_per_frame <= 0 or min_frames < 1:
        raise ValueError("Invalid frames or budget")
    count = min(len(frames), token_budget // tokens_per_frame)
    if count < min_frames:
        raise ValueError("Budget cannot fit minimum frames")
    if count == len(frames):
        return list(range(count))
    coverage = max(1, count // 2)
    chosen = set(torch.linspace(0, len(frames) - 1, coverage).round().long().tolist())
    scores = torch.zeros(len(frames), device=frames.device)
    scores[1:] = (frames[1:].float() - frames[:-1].float()).abs().mean((1, 2, 3))
    for index in sorted(range(len(frames)), key=lambda i: (-float(scores[i]), i)):
        if len(chosen) == count:
            break
        chosen.add(index)
    return sorted(chosen)
