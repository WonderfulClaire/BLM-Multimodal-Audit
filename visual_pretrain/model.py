"""Compact vision/text dual tower for executable experiments; not pretrained Qwen or BGE weights."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_resolution import QwenViT


# =====================================================================
# 简化版 Text Encoder（BGE-like）
# =====================================================================
class SimpleTextEncoder(nn.Module):
    """简化版 BGE-style Text Encoder。

    笔记原文：
      "目前文本表征主要使用 BGE 模型，该模型类似于 BERT，只能表征 512 个token 的文本，
       原因是其基于可训练的位置编码（不是基于旋转位置编码或者正余弦位置编码），外推性较差"

    NOTE: 真生产请直接 load BGE：
        from transformers import AutoModel
        model = AutoModel.from_pretrained("BAAI/bge-large-zh-v1.5")
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 384,
        max_len: int = 64,
        num_layers: int = 2,
        num_heads: int = 6,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)  # 笔记："基于可训练的位置编码"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, L) → returns: (B, embed_dim) text embedding."""
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.encoder(x, src_key_padding_mask=input_ids.eq(0))
        # 池化：用 [CLS] token（input_ids 第 0 个 token）的 embedding
        return x[:, 0, :]


# =====================================================================
# 完整的 CLIP 模型（Image + Text 双塔）
# =====================================================================
class BLMCLIP(nn.Module):
    """BLM（风控大模型）的视觉-文本双塔模型。

    对应笔记："对比学习，提升语义检索能力"
    """

    def __init__(
        self,
        embed_dim: int = 256,
        img_embed_dim: int = 384,
        text_embed_dim: int = 384,
        img_backbone_kwargs: Optional[dict] = None,
        text_backbone_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        img_backbone_kwargs = img_backbone_kwargs or {}
        text_backbone_kwargs = text_backbone_kwargs or {}

        self.image_tower = QwenViT(embed_dim=img_embed_dim, **img_backbone_kwargs)
        self.text_tower = SimpleTextEncoder(
            embed_dim=text_embed_dim, **text_backbone_kwargs
        )
        # 投影头（CLIP 里这两个投影到共享维度）
        self.image_proj = nn.Linear(img_embed_dim, embed_dim)
        self.text_proj = nn.Linear(text_embed_dim, embed_dim)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, C, H, W) → (B, embed_dim) L2-normalized。"""
        feat = self.image_tower(images)
        feat = self.image_proj(feat)
        return F.normalize(feat, dim=-1)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, L) → (B, embed_dim) L2-normalized。"""
        feat = self.text_tower(input_ids)
        feat = self.text_proj(feat)
        return F.normalize(feat, dim=-1)

    def forward(self, images, text_ids):
        return self.encode_image(images), self.encode_text(text_ids)


if __name__ == "__main__":
    model = BLMCLIP(embed_dim=128, img_embed_dim=384, text_embed_dim=384)
    img = torch.randn(2, 3, 224, 224)
    txt = torch.randint(0, 1000, (2, 32))
    img_emb, txt_emb = model(img, txt)
    print(f"[BLMCLIP] img_emb: {img_emb.shape}, txt_emb: {txt_emb.shape}")
