"""visual_pretrain/__init__.py"""

from .dynamic_resolution import (
    SmartResizeConfig,
    QwenViT,
    WindowAttention,
    MRotaryEmbedding2D,
)
from .dynamic_frames import (
    VideoBinPackerConfig,
    video_binpack,
    VideoEncoder,
    VideoSample,
)
from .binpack import (
    OfflineBinPackerConfig,
    OnlineBinPackerConfig,
    OnlineBinPacker,
    Bin,
    ImageSample,
    offline_binpack,
)
from .dist_sync import (
    RingAllGather,
    BidirAllGather,
    CLIPContrastiveLoss,
    sync_dynamic_batch_size,
)
from .fg_clip import (
    FGClipLoss,
    ReferringExpressionExtractor,
    BBoxGenerator,
    HardNegativeBuilder,
)
from .model import BLMCLIP, SimpleTextEncoder

__all__ = [
    "SmartResizeConfig",
    "QwenViT",
    "WindowAttention",
    "MRotaryEmbedding2D",
    "VideoBinPackerConfig",
    "video_binpack",
    "VideoEncoder",
    "VideoSample",
    "OfflineBinPackerConfig",
    "OnlineBinPackerConfig",
    "OnlineBinPacker",
    "Bin",
    "ImageSample",
    "offline_binpack",
    "RingAllGather",
    "BidirAllGather",
    "CLIPContrastiveLoss",
    "sync_dynamic_batch_size",
    "FGClipLoss",
    "ReferringExpressionExtractor",
    "BBoxGenerator",
    "HardNegativeBuilder",
    "BLMCLIP",
    "SimpleTextEncoder",
]
