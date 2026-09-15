"""Global, region and reviewed hard-negative contrastive objectives. Grounding requires a real backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 数据结构
# =====================================================================
@dataclass
class RegionSample:
    """一个区域样本：图像中的一个区域 + 多个文本表述。"""

    image_id: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    positive_texts: List[str]  # 正样本文本（多个）
    hard_neg_texts: List[str]  # 难负例文本
    # 注意：难负例是基于正样本通过"属性/动作改写"得到的


# =====================================================================
# 1) FG-CLIP 三损失
# =====================================================================
class FGClipLoss(nn.Module):
    """FG-CLIP 三损失求和。

    笔记原文：
      "全局对齐、局部对齐、难负例对齐，loss就是三个loss的和"
      "CLIP 损失函数：全局损失函数是图片和文本的相似度..."
      "区域对比学习：本质与 CLIP 损失函数类似..."
      "难例处理：对于同一区域的多个文本表述，希望与第一个文本相似度最大，与其他文本相似度最小"

    Args:
        global_weight: 全局 CLIP 损失权重
        region_weight: 区域对比损失权重
        hard_neg_weight: 难负例损失权重
        temperature: 温度参数
    """

    def __init__(
        self,
        global_weight: float = 1.0,
        region_weight: float = 1.0,
        hard_neg_weight: float = 1.0,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.global_weight = global_weight
        self.region_weight = region_weight
        self.hard_neg_weight = hard_neg_weight
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def _info_nce(
        self,
        anchor: torch.Tensor,  # (B, D)
        positive: torch.Tensor,  # (B, D)
        negatives: Optional[torch.Tensor] = None,  # (B, K, D) or (B*N, D)
    ) -> torch.Tensor:
        """通用 InfoNCE 损失。

        相似度 = anchor @ positive^T  → softmax → 交叉熵，目标是 anchor 与自己 positive 相似度最大。
        """
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * anchor @ positive.T  # (B, B)

        if negatives is not None:
            # 把负例拼接进 logits
            neg_logits = (
                scale * anchor @ negatives.reshape(-1, negatives.shape[-1]).T
            )  # (B, B*K)
            logits = torch.cat([logits, neg_logits], dim=-1)

        labels = torch.arange(anchor.shape[0], device=anchor.device)
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        global_image_emb: torch.Tensor,  # (B, D) — 全图 embedding
        global_text_emb: torch.Tensor,  # (B, D) — 全局 caption embedding
        region_image_emb: torch.Tensor,  # (R, D) — 每个 region 的 embedding
        region_text_emb: torch.Tensor,  # (R, D) — region 对应正样本 embedding
        hard_neg_emb: Optional[
            torch.Tensor
        ] = None,  # (R, K, D) — 每个 region 的难负例 embedding
        first_positive_idx: int = 0,  # 难负例损失里"第一个正样本"在 region_text_emb 中的索引
    ) -> Tuple[torch.Tensor, dict]:
        """Returns: (total_loss, loss_components_dict)。"""
        # === 全局对齐损失 ===
        loss_global = (
            self._info_nce(global_image_emb, global_text_emb)
            + self._info_nce(global_text_emb, global_image_emb)
        ) / 2

        # === 局部对齐损失（region ↔ referring expression）===
        loss_region = torch.tensor(0.0, device=global_image_emb.device)
        if region_image_emb.shape[0] > 0:
            loss_region = (
                self._info_nce(region_image_emb, region_text_emb)
                + self._info_nce(region_text_emb, region_image_emb)
            ) / 2

        # === 难负例损失 ===
        # 笔记原文：
        #   "对于同一区域的多个文本表述，希望与第一个文本相似度最大，与其他文本相似度最小"
        # 即：每个 region 对应 1 个正样本 + K 个难负例
        # loss = -log(exp(s(r, t_pos)) / (exp(s(r, t_pos)) + sum_k exp(s(r, t_neg_k))))
        loss_hard_neg = torch.tensor(0.0, device=global_image_emb.device)
        if hard_neg_emb is not None and region_image_emb.shape[0] > 0:
            R, K, D = hard_neg_emb.shape
            scale = self.logit_scale.exp().clamp(max=100.0)
            # region_image_emb: (R, D)；positive 用 region_text_emb 的第 first_positive_idx 行（每个 region 都用自己的正样本）
            # 这里简化：hard_neg_emb 和 region_image_emb 一一对应
            pos_sim = (
                region_image_emb
                * region_text_emb[first_positive_idx : first_positive_idx + R]
            ).sum(dim=-1, keepdim=True)  # (R, 1)
            neg_sim = (region_image_emb.unsqueeze(1) * hard_neg_emb).sum(
                dim=-1
            )  # (R, K)
            logits = scale * torch.cat([pos_sim, neg_sim], dim=-1)  # (R, K+1)
            labels = torch.zeros(
                R, dtype=torch.long, device=logits.device
            )  # 正样本在第 0 列
            loss_hard_neg = F.cross_entropy(logits, labels)

        total = (
            self.global_weight * loss_global
            + self.region_weight * loss_region
            + self.hard_neg_weight * loss_hard_neg
        )

        return total, {
            "loss_global": loss_global.item()
            if torch.is_tensor(loss_global)
            else loss_global,
            "loss_region": loss_region.item()
            if torch.is_tensor(loss_region)
            else loss_region,
            "loss_hard_neg": loss_hard_neg.item()
            if torch.is_tensor(loss_hard_neg)
            else loss_hard_neg,
        }


# =====================================================================
# 2) SpaCy 引用表达式提取（stub）
# =====================================================================
class ReferringExpressionExtractor:
    """Compatibility adapter for the actual spaCy backend."""

    def __init__(self, model_name="en_core_web_sm"):
        from .pipeline import SpacyExtractor

        self.backend = SpacyExtractor(model_name)

    def extract(self, caption):
        return self.backend(caption)


class BBoxGenerator:
    """YOLO-World 边界框生成（stub）。

    笔记原文：
      "具体获取边界框的方式是使用了 Yolo-World。YOLO-World 是一种结合了 YOLO 和 开放词汇能力的实时目标检测模型"

    NOTE: 这是 stub，生产请用：
      from ultralytics import YOLOWorld
      model = YOLOWorld("yolov8s-world.pt")
      results = model.predict(image_path)
      # results[0].boxes.xyxy, results[0].boxes.conf
    """

    def __init__(self, confidence_threshold: float = 0.4):
        self.conf_threshold = confidence_threshold

    def generate(self, image_id, referring_expressions, *, detector=None):
        """Run an explicitly supplied grounding backend; never invent boxes."""
        if detector is None:
            raise ValueError(
                "Supply a grounding backend or use visual_pretrain.pipeline cached mode"
            )
        from PIL import Image

        return detector(Image.open(image_id).convert("RGB"), referring_expressions)

    def nms(
        self,
        bboxes: List[Tuple[float, float, float, float]],
        scores: List[float],
        iou_threshold: float = 0.5,
    ) -> List[int]:
        """非极大值抑制。

        笔记原文：
          "只保留置信度得分高于 0.4 的边界框...非极大值抑制会保留得分最高的那一个边界框，
           并移除所有与它高度重叠的其他边界框"
        """
        # 1) 过滤低置信度
        keep_indices = [i for i, s in enumerate(scores) if s >= self.conf_threshold]
        if not keep_indices:
            return []

        # 2) 按 score 降序
        keep_indices = sorted(keep_indices, key=lambda i: -scores[i])

        def iou(b1, b2):
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            return inter / (area1 + area2 - inter + 1e-9)

        final_keep = []
        while keep_indices:
            best = keep_indices.pop(0)
            final_keep.append(best)
            keep_indices = [
                i for i in keep_indices if iou(bboxes[best], bboxes[i]) < iou_threshold
            ]
        return final_keep


# =====================================================================
# 4) 难负例构造（属性/动作改写）
# =====================================================================
class HardNegativeBuilder:
    """难负例生成：通过属性/动作改写构造难负例。

    笔记原文：
      "这里的难负例的创建实际上是通过改写正样本中的文本实现的"
      "基于文生实体抽取的文本。通过大模型生成相似表述作为难负例，保留核心对象，修改属性和动作"
      "负例和正例在空间上很类似，但在人看来完全不同"

    NOTE: 这是 stub，生产请用 Qwen-VL / DeepSeek 等大模型做改写。
    """

    # 简单的属性词典（实际要全得多，可以从 wordnet 加载）
    COLOR_REPLACE = {
        "红色": "蓝色",
        "蓝色": "红色",
        "黑色": "白色",
        "白色": "黑色",
        "绿色": "黄色",
        "黄色": "绿色",
    }
    ACTION_REPLACE = {
        "跳跃": "奔跑",
        "奔跑": "跳跃",
        "坐": "站",
        "站": "坐",
    }

    def build(self, positive_text: str) -> str:
        """对正样本做"属性/动作"改写，得到难负例。"""
        # Exactly one substitution: do not replace blue back to red in the same pass.
        for mapping in (self.COLOR_REPLACE, self.ACTION_REPLACE):
            for old, new in mapping.items():
                if old in positive_text:
                    return positive_text.replace(old, new, 1)
        raise ValueError(
            "No supported attribute/action to change; request a reviewed candidate"
        )


# =====================================================================
# Smoke test
# =====================================================================
