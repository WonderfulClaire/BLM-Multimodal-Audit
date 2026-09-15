"""
grpo_post_training/policy.py
============================
对应 Obsidian 笔记章节：
  §四 §4  MLLM Post-training（Policy Model = MLLM，全参微调）

本文件实现：
  - 一个简化版 MLLM Policy：ViT encoder + 小型 LM head，能接收图像+文本 prompt，
    输出 JSON 格式的审核结果 {risk_category, risk_level, reason}

  NOTE: 真生产请直接 load Qwen2.5-VL：
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from visual_pretrain.dynamic_resolution import QwenViT


# =====================================================================
# 数据结构
# =====================================================================
@dataclass
class AuditOutput:
    """审核输出：JSON 格式 {risk_category, risk_level, reason}。"""

    risk_category: str
    risk_level: str
    reason: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


# =====================================================================
# 简化版 MLLM Policy
# =====================================================================
class SimpleMLLMPolicy(nn.Module):
    """简化版 MLLM Policy 模型。

    结构：
      - ViT (QwenViT) —— 视觉编码
      - Projector —— 把视觉 token 投到 text embedding 空间
      - TextDecoder —— 简化版 autoregressive LM（GPT 风格）

    输入：image (B, C, H, W) + text_ids (B, L)
    输出：logits (B, L, V) —— 下一个 token 的 logits
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        img_size: int = 224,
        patch_size: int = 28,
        img_embed_dim: int = 384,
        text_embed_dim: int = 384,
        text_num_layers: int = 4,
        text_num_heads: int = 6,
        max_text_len: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len

        # Vision
        self.vit = QwenViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=img_embed_dim,
            depth=4,
            num_heads=6,
            window_size=4,
        )
        self.vision_proj = nn.Linear(img_embed_dim, text_embed_dim)

        # Text
        self.token_emb = nn.Embedding(vocab_size, text_embed_dim)
        self.pos_emb = nn.Embedding(max_text_len, text_embed_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=text_embed_dim,
            nhead=text_num_heads,
            dim_feedforward=text_embed_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=text_num_layers)
        self.lm_head = nn.Linear(text_embed_dim, vocab_size)

    def forward(
        self,
        images: torch.Tensor,  # (B, C, H, W)
        text_ids: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B, L, V) —— 每个位置预测下一个 token
        """

        B = images.shape[0]
        device = images.device
        # 1) 视觉编码
        vision_feat = self.vit(images)  # (B, img_embed_dim)
        vision_feat = self.vision_proj(vision_feat)  # (B, text_embed_dim)
        # 扩展成 (B, 1, text_embed_dim) 作为"图像 token"
        vision_tokens = vision_feat.unsqueeze(1)  # (B, 1, D)

        # 2) 文本编码
        L = text_ids.shape[1]
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        text_emb = self.token_emb(text_ids) + self.pos_emb(positions)  # (B, L, D)

        # 3) 拼接 vision tokens + text tokens
        combined = torch.cat([vision_tokens, text_emb], dim=1)  # (B, 1+L, D)

        # 4) 解码
        causal = torch.triu(
            torch.ones(
                combined.shape[1], combined.shape[1], device=device, dtype=torch.bool
            ),
            diagonal=1,
        )
        hidden = self.decoder(combined, mask=causal)  # (B, 1+L, D)

        # 5) LM head
        logits = self.lm_head(hidden)  # (B, 1+L, V)

        # 只返回对应文本位置的 logits
        return logits[:, 1:, :]  # (B, L, V)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,  # (B, L_prompt)
        max_new_tokens: int = 64,
        temperature: float = 0.7,
    ) -> torch.Tensor:
        """Greedy/temperature sampling 自回归生成。

        Returns:
            generated_ids: (B, L_prompt + new_tokens)
        """

        # 把 prompt_ids 转成 embedding + 视觉 token，一次过 decoder
        # 然后逐 token 生成
        cur_ids = prompt_ids
        for _ in range(max_new_tokens):
            logits = self.forward(images, cur_ids)  # (B, cur_len, V)
            next_logits = logits[:, -1, :] / temperature  # (B, V)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            cur_ids = torch.cat([cur_ids, next_token], dim=1)
        return cur_ids


# =====================================================================
# 输出解析：模型生成的字符串 → AuditOutput
# =====================================================================
def parse_audit_output(text: str) -> Optional[AuditOutput]:
    """从模型生成的字符串中解析 JSON。"""
    # 尝试匹配 {...} 形式
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return AuditOutput(
            risk_category=data.get("risk_category", ""),
            risk_level=data.get("risk_level", ""),
            reason=data.get("reason", ""),
        )
    except json.JSONDecodeError:
        return None


# =====================================================================
# Smoke test
# =====================================================================
if __name__ == "__main__":
    model = SimpleMLLMPolicy(vocab_size=1000, img_embed_dim=384, text_embed_dim=384)
    img = torch.randn(2, 3, 224, 224)
    txt = torch.randint(0, 1000, (2, 32))
    logits = model(img, txt)
    print(f"[SimpleMLLMPolicy] logits shape: {logits.shape}")

    # 生成
    prompt = torch.randint(0, 1000, (1, 8))
    generated = model.generate(img[:1], prompt, max_new_tokens=10)
    print(
        f"[Generate] prompt shape {prompt.shape} -> generated shape {generated.shape}"
    )

    # 解析
    fake_text = '{"risk_category": "暴力", "risk_level": "高危", "reason": "画面血腥"}'
    parsed = parse_audit_output(fake_text)
    print(f"[ParseAudit] {parsed.to_json()}")
