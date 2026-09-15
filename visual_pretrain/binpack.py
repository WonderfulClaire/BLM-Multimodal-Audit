"""Token-budget first-fit-decreasing scheduling. Waste is measured per output bin; no throughput guarantee."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


# =====================================================================
# Image Token 计数工具
# =====================================================================
def default_token_counter(image_meta) -> int:
    """根据 smart resize 后的 (H, W) 算 token 数。

    笔记原文：
      "Token数是怎么算的：ViT流程：首先图片经过一个smart resize，把图片过一层卷积核，
       卷积核的大小可能是28×28，最后切出来的token数实际上就是你行的像素除上28，
       再乘上纵的像素数除上28"

    image_meta: (height, width)
    """
    h, w = image_meta
    return (h // 28) * (w // 28)


# =====================================================================
# 1) OfflineBinPacker（开源数据）
# =====================================================================
@dataclass
class OfflineBinPackerConfig:
    bin_size: int = 4096  # 一个 bin 的总 token 上限
    shuffle_seed: int = 42
    image_token_counter: Callable = default_token_counter
    # 输出格式：每个 bin 一个 dict，记录 bin 内图片路径 / token 数
    save_tar: bool = (
        True  # 笔记："打包好的一个一个bin是以tar的方式去呈现到文件系统中的"
    )


@dataclass
class ImageSample:
    image_id: str
    meta: tuple  # (H, W) smart resize 后的尺寸
    path: str = ""  # 实际图片路径
    token_count: int = field(default=0)

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = default_token_counter(self.meta)


@dataclass
class Bin:
    bin_id: str
    samples: List[ImageSample] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(s.token_count for s in self.samples)

    @property
    def waste_ratio(self) -> float:
        used = self.total_tokens
        return (self.cfg_max - used) / self.cfg_max

    # 后面单独 cfg_max 由 packer 注入
    cfg_max: int = 4096


def offline_binpack(
    samples: List[ImageSample],
    cfg: Optional[OfflineBinPackerConfig] = None,
) -> List[Bin]:
    """开源数据离线 binpack 算法（first-fit-decreasing, FFD）。

    步骤：
      1. shuffle 让图片分布均匀（笔记："对数据整体shuffle使分布均匀"）
      2. 按 token 数降序排
      3. FFD：依次塞进剩余空间 >= 当前图片 token 数的第一个 bin
      4. 没有就开新 bin
    """
    cfg = cfg or OfflineBinPackerConfig()
    rng = random.Random(cfg.shuffle_seed)

    # 1) shuffle
    samples = list(samples)
    if cfg.bin_size <= 0 or any(
        s.token_count <= 0 or s.token_count > cfg.bin_size for s in samples
    ):
        raise ValueError("Image token count must fit positive bin budget")
    rng.shuffle(samples)

    # 2) 按 token 数降序
    samples = sorted(samples, key=lambda s: -s.token_count)

    bins: List[Bin] = []

    for sample in samples:
        placed = False
        # 尝试装进第一个剩余空间 >= sample.token_count 的 bin
        for b in bins:
            if b.total_tokens + sample.token_count <= cfg.bin_size:
                b.samples.append(sample)
                placed = True
                break
        if not placed:
            new_bin = Bin(bin_id=f"bin_{len(bins):04d}.tar")
            new_bin.samples.append(sample)
            bins.append(new_bin)

    # 过滤掉浪费率过高的 bin（demo 阶段不过滤；生产按业务设阈值）
    # NOTE: 笔记里说"控制在 max_waste 之内"，但 demo 数据量小时 1-2 个 bin 的浪费率自然会偏高。
    # 这里只在 waste > bin_size 才过滤（避免空 bin）。
    bins = [b for b in bins if b.total_tokens <= cfg.bin_size]

    # 更新 cfg_max
    for b in bins:
        b.cfg_max = cfg.bin_size

    return bins


# =====================================================================
# 2) OnlineBinPacker（业务数据）
# =====================================================================
@dataclass
class OnlineBinPackerConfig:
    bin_size: int = 4096
    max_image_size: tuple = (
        392,
        224,
    )  # 笔记："业务数据集，image size 最大为 (392, 224)"
    image_token_counter: Callable = default_token_counter


class OnlineBinPacker:
    """业务数据在线 binpack。

    笔记原文：
      "对于业务数据集，image size 最大为 (392, 224)，直接采用在线组 bin 的方式进行 bin pack，
       主要是通过设置较大的 bin，然后通过添加 image 直至 bin 上限的方式实现"

    因为业务数据 size 有上限（392x224），所以可以直接贪心：
      一个一个 image 加进当前 bin，直到 next image 装不下就开新 bin。
      最大浪费率：bin=4096，最大 image=(392,224) → token=14*8=112 → 112/4096=2.7%
    """

    def __init__(self, cfg: Optional[OnlineBinPackerConfig] = None):
        self.cfg = cfg or OnlineBinPackerConfig()
        self.current_bin: List[ImageSample] = []
        self.current_used: int = 0
        self.bins: List[Bin] = []

    def add(self, sample: ImageSample) -> None:
        """加入一张图。若放不下当前 bin 就把当前 bin 提交，再开新 bin。"""
        if sample.token_count <= 0 or sample.token_count > self.cfg.bin_size:
            raise ValueError("Image token count must fit positive bin budget")
        if self.current_used + sample.token_count <= self.cfg.bin_size:
            self.current_bin.append(sample)
            self.current_used += sample.token_count
        else:
            # 提交当前 bin
            self._commit()
            # 开新 bin
            self.current_bin = [sample]
            self.current_used = sample.token_count

    def _commit(self) -> None:
        if not self.current_bin:
            return
        bin = Bin(
            bin_id=f"online_bin_{len(self.bins):04d}",
            samples=self.current_bin,
            cfg_max=self.cfg.bin_size,
        )
        self.bins.append(bin)
        self.current_bin = []
        self.current_used = 0

    def finish(self) -> List[Bin]:
        """训练结束调用一次，提交最后一个 bin。"""
        self._commit()
        return self.bins


# =====================================================================
# Smoke test
# =====================================================================
if __name__ == "__main__":
    # 1) Offline
    np.random.seed(0)
    offline_samples = [
        ImageSample(
            image_id=f"img_{i:04d}",
            meta=(np.random.randint(28, 800), np.random.randint(28, 800)),
        )
        for i in range(50)
    ]
    cfg_off = OfflineBinPackerConfig(bin_size=4096)
    bins_off = offline_binpack(offline_samples, cfg_off)
    print(f"[OfflineBinPack] {len(offline_samples)} imgs -> {len(bins_off)} bins")
    for i, b in enumerate(bins_off[:3]):
        print(
            f"  bin {i}: {len(b.samples)} imgs, total={b.total_tokens}/{cfg_off.bin_size}, waste={b.waste_ratio:.2%}"
        )

    # 2) Online
    online_packer = OnlineBinPacker()
    # 模拟训练时图片一个个来，size 最大 392x224
    for i in range(20):
        h = np.random.randint(28, 392)
        w = np.random.randint(28, 224)
        online_packer.add(ImageSample(image_id=f"img_{i:04d}", meta=(h, w)))
    bins_on = online_packer.finish()
    print(f"\n[OnlineBinPack] 20 imgs -> {len(bins_on)} bins")
    for i, b in enumerate(bins_on):
        print(
            f"  bin {i}: {len(b.samples)} imgs, total={b.total_tokens}/4096, waste={b.waste_ratio:.2%}"
        )
