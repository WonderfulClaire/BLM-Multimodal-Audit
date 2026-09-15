"""
grpo_post_training/reward.py
=============================
对应 Obsidian 笔记章节：
  §四 §4 §2  Reward Model 的设计（5 条原则）

本文件实现 5 个 Reward 维度 + 一个加权汇总：

  1. Recall on Critical Risks       —— 风险召回率
  2. Category Hit                    —— 风险类别命中
  3. Consistency of Reasoning        —— 判责理由一致性
  4. Format Adherence                —— 结构化格式遵循
  5. Complex Instruction Following   —— 复杂指令理解

设计动机（讲给面试官听）：
  GRPO 用"是否正确回答审核规则"作为准确性奖励，单维 reward 容易让模型走偏（reward hacking）。
  拆成 5 维 + 加权，能让模型同时兼顾召回 / 命中 / 一致 / 格式 / 指令理解。
  每条原则都配了反例（reward 应该低的）和正例（reward 应该高的），方便讲清楚。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .policy import AuditOutput, parse_audit_output


# =====================================================================
# 数据结构
# =====================================================================
@dataclass
class AuditSample:
    """单条审核样本：图像 + 真实标签 + 模型输出。"""

    image_id: str
    prompt: str
    ground_truth: AuditOutput  # 真实标注
    model_output: str  # 模型生成的字符串
    parsed_output: Optional[AuditOutput] = None  # parse 后的结构化输出
    critical_risks: List[str] = field(default_factory=list)  # 高危风险列表


@dataclass
class RewardBreakdown:
    """5 维 reward 分数。"""

    recall: float = 0.0
    category_hit: float = 0.0
    consistency: float = 0.0
    format: float = 0.0
    instruction: float = 0.0

    @property
    def total(self) -> float:
        # 笔记里没明确权重，这里给默认权重（求和 + 等权重的折中）
        return (
            0.3 * self.recall
            + 0.25 * self.category_hit
            + 0.2 * self.consistency
            + 0.15 * self.format
            + 0.1 * self.instruction
        )


# =====================================================================
# 5 个 Reward 维度
# =====================================================================
class RewardModel:
    """5 维 Reward Model。"""

    # 复杂指令模式（"仅穿吊带不算色情"这种）
    INSTRUCTION_PATTERNS = [
        r"仅穿.+\u4e0d算",  # "仅穿吊带不算色情"
        r"\u53ea\u6709.+才算",  # "只有露点才算色情"
        r"\u9664\u975e.+",  # "除非..." 例外条款
    ]

    def __init__(
        self,
        # 各维度权重（默认等权）
        w_recall: float = 0.3,
        w_category_hit: float = 0.25,
        w_consistency: float = 0.2,
        w_format: float = 0.15,
        w_instruction: float = 0.1,
    ):
        self.w_recall = w_recall
        self.w_category_hit = w_category_hit
        self.w_consistency = w_consistency
        self.w_format = w_format
        self.w_instruction = w_instruction

    def score(self, sample: AuditSample) -> RewardBreakdown:
        """计算单条样本的 5 维 reward。"""
        if sample.parsed_output is None:
            sample.parsed_output = parse_audit_output(sample.model_output)

        rb = RewardBreakdown()

        # === 1) 风险召回率（Recall on Critical Risks）===
        # 笔记原文：
        #   "漏掉高危风险（如自杀自残、涉政涉恐）是重大错误，包含此类漏报的输出排名应被置于最低。"
        #   "视频包含一个一闪而过的自残画面。输出A：'视频内容正常。' 输出B：'视频内容正常，但情绪略显低落。'
        #    排序：B > A（A 和 B 因为漏报了关键风险，排名急剧下降）"
        rb.recall = self._score_recall(sample)

        # === 2) 风险类别命中（Category Hit）===
        # 笔记原文：
        #   "命中平台定义的真实风险类别越多，排名越高"
        rb.category_hit = self._score_category_hit(sample)

        # === 3) 判责理由一致性（Consistency of Reasoning）===
        # 笔记原文：
        #   "给出的风险等级必须和识别出的风险类别及内容相匹配"
        #   "输出B：{risk_category: '文本违规', risk_level: '高危', reason: '视频画面血腥'}
        #    排序：A > B（B 的判责理由和类别不一致）"
        rb.consistency = self._score_consistency(sample)

        # === 4) 结构化格式遵循（Format Adherence）===
        # 笔记原文：
        #   "严格遵循预定义的输出格式（如 JSON），格式正确、无缺失字段的输出，排名更高"
        rb.format = self._score_format(sample)

        # === 5) 复杂指令理解（Complex Instruction Following）===
        # 笔记原文：
        #   "能理解并执行复杂的风控指令。例如'仅穿吊带不算色情'"
        #   "输出A：视频出现擦边色情片段  输出B：视频中人物虽然穿了吊带但是不属于色情  排序：B > A"
        rb.instruction = self._score_instruction(sample)

        return rb

    def _score_recall(self, sample: AuditSample) -> float:
        """风险召回率：是否漏报 critical risks。"""
        if not sample.critical_risks:
            # 没有 critical risks，直接给满分（不漏就行）
            return 1.0
        if sample.parsed_output is None:
            return 0.0  # 没 parse 出来 = 漏报
        # 检查 ground_truth 是否包含了 critical_risks
        gt_cats = sample.ground_truth.risk_category
        # 简化：ground_truth 里提到了 critical_risks 中的任一，就算召回
        for cr in sample.critical_risks:
            if cr in gt_cats:
                # ground truth 提到了 critical risk，模型也应该提到
                if (
                    cr in sample.parsed_output.risk_category
                    or cr in sample.parsed_output.reason
                ):
                    continue  # OK
                else:
                    return 0.0  # 漏报 → 0 分
        return 1.0

    def _score_category_hit(self, sample: AuditSample) -> float:
        """风险类别命中：命中真实风险类别越多越好。"""
        if sample.parsed_output is None:
            return 0.0
        gt_cat = sample.ground_truth.risk_category
        pred_cat = sample.parsed_output.risk_category
        # 简化判定：字符串包含关系
        if gt_cat == pred_cat:
            return 1.0
        if gt_cat in pred_cat or pred_cat in gt_cat:
            return 0.5
        return 0.0

    def _score_consistency(self, sample: AuditSample) -> float:
        """判责理由一致性：reason 必须和 category + level 匹配。"""
        if sample.parsed_output is None:
            return 0.0
        # 简化判定：如果 reason 包含 category 的关键词 + level 关键词，给 1.0；否则 0.5
        pred_cat = sample.parsed_output.risk_category
        pred_level = sample.parsed_output.risk_level
        reason = sample.parsed_output.reason
        # 反例："category=文本违规, level=高危, reason=视频画面血腥" → 不一致
        if "血腥" in reason and pred_cat != "暴力":
            return 0.0
        if pred_cat and pred_cat[:2] in reason:
            return 1.0
        if pred_level and pred_level in reason:
            return 0.8
        return 0.5

    def _score_format(self, sample: AuditSample) -> float:
        """结构化格式遵循：是否合法 JSON。"""
        if sample.parsed_output is None:
            return 0.0
        # 进一步检查必填字段
        if not all(
            [
                sample.parsed_output.risk_category,
                sample.parsed_output.risk_level,
                sample.parsed_output.reason,
            ]
        ):
            return 0.5
        return 1.0

    def _score_instruction(self, sample: AuditSample) -> float:
        """复杂指令理解：是否理解 prompt 里的例外条款。"""
        if sample.parsed_output is None:
            return 0.0
        # 检查 prompt 是否有 instruction pattern
        for pat in self.INSTRUCTION_PATTERNS:
            if re.search(pat, sample.prompt):
                # prompt 含例外条款，模型应该"不判定"或"低风险"
                if (
                    "不算" in sample.parsed_output.reason
                    or "不属于" in sample.parsed_output.reason
                ):
                    return 1.0
                if sample.parsed_output.risk_level in ["低危", "无", ""]:
                    return 0.8
                return 0.0
        # 没有 instruction pattern，默认满分
        return 1.0


# =====================================================================
# 测试样例（对应笔记里的 5 个示例）
# =====================================================================
def demo_examples() -> None:
    """跑一遍笔记里给的 5 个示例，看 reward 怎么打。"""
    rm = RewardModel()

    # 示例 1：风险召回率（笔记里"B > A"那个例）
    s1a = AuditSample(
        image_id="v1",
        prompt="审核视频...",
        ground_truth=AuditOutput("自残", "高危", "画面包含自残内容"),
        model_output="视频内容正常。",
        critical_risks=["自残"],
    )
    s1b = AuditSample(
        image_id="v1",
        prompt="审核视频...",
        ground_truth=AuditOutput("自残", "高危", "画面包含自残内容"),
        model_output="视频内容正常，但情绪略显低落。",
        critical_risks=["自残"],
    )
    rb1a = rm.score(s1a)
    rb1b = rm.score(s1b)
    print(
        f"[Recall] A={rb1a.total:.2f}, B={rb1b.total:.2f}  (笔记：B > A ✓)"
        if rb1b.total > rb1a.total
        else "[Recall] WRONG"
    )

    # 示例 2：风险类别命中
    s2a = AuditSample(
        image_id="v2",
        prompt="审核视频...",
        ground_truth=AuditOutput("低俗内容+涉赌", "高危", "低俗着装 + 宣传赌博网站"),
        model_output='{"risk_category": "低俗内容", "risk_level": "高危", "reason": "低俗着装"}',
    )
    s2b = AuditSample(
        image_id="v2",
        prompt="审核视频...",
        ground_truth=AuditOutput("低俗内容+涉赌", "高危", "低俗着装 + 宣传赌博网站"),
        model_output='{"risk_category": "低俗内容和涉赌风险", "risk_level": "高危", "reason": "低俗着装 + 赌博网站"}',
    )
    rb2a = rm.score(s2a)
    rb2b = rm.score(s2b)
    print(
        f"[Category] A={rb2a.category_hit:.2f}, B={rb2b.category_hit:.2f}  (B > A ✓)"
        if rb2b.category_hit > rb2a.category_hit
        else "[Category] WRONG"
    )

    # 示例 3：判责理由一致性
    s3a = AuditSample(
        image_id="v3",
        prompt="审核视频...",
        ground_truth=AuditOutput("文本违规", "高危", "侮辱性词汇"),
        model_output='{"risk_category": "文本违规", "risk_level": "高危", "reason": "视频包含侮辱性词汇"}',
    )
    s3b = AuditSample(
        image_id="v3",
        prompt="审核视频...",
        ground_truth=AuditOutput("文本违规", "高危", "侮辱性词汇"),
        model_output='{"risk_category": "文本违规", "risk_level": "高危", "reason": "视频画面血腥"}',
    )
    rb3a = rm.score(s3a)
    rb3b = rm.score(s3b)
    print(
        f"[Consistency] A={rb3a.consistency:.2f}, B={rb3b.consistency:.2f}  (A > B ✓)"
        if rb3a.consistency > rb3b.consistency
        else "[Consistency] WRONG"
    )

    # 示例 4：格式遵循
    s4a = AuditSample(
        image_id="v4",
        prompt="审核视频...",
        ground_truth=AuditOutput("暴力", "高危", "暴力内容"),
        model_output='{"risk": "暴力", "level": "高危"}',
    )
    s4b = AuditSample(
        image_id="v4",
        prompt="审核视频...",
        ground_truth=AuditOutput("暴力", "高危", "暴力内容"),
        model_output="风险类别是暴力，风险等级是高危。",
    )
    rb4a = rm.score(s4a)
    rb4b = rm.score(s4b)
    print(
        f"[Format] A={rb4a.format:.2f}, B={rb4b.format:.2f}  (A > B ✓)"
        if rb4a.format > rb4b.format
        else "[Format] WRONG"
    )

    # 示例 5：复杂指令
    s5a = AuditSample(
        image_id="v5",
        prompt="仅穿吊带不算色情，审核视频...",
        ground_truth=AuditOutput("色情", "低危", "穿了吊带不属于色情"),
        model_output='{"risk_category": "色情", "risk_level": "高危", "reason": "擦边色情片段"}',
    )
    s5b = AuditSample(
        image_id="v5",
        prompt="仅穿吊带不算色情，审核视频...",
        ground_truth=AuditOutput("色情", "低危", "穿了吊带不属于色情"),
        model_output='{"risk_category": "色情", "risk_level": "低危", "reason": "穿了吊带但是不属于色情"}',
    )
    rb5a = rm.score(s5a)
    rb5b = rm.score(s5b)
    print(
        f"[Instruction] A={rb5a.instruction:.2f}, B={rb5b.instruction:.2f}  (B > A ✓)"
        if rb5b.instruction > rb5a.instruction
        else "[Instruction] WRONG"
    )


if __name__ == "__main__":
    demo_examples()
