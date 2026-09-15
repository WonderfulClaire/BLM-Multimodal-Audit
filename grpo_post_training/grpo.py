"""PPO/GRPO objectives, group advantages and masked KL. No measured speed claims."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# GAE（Generalized Advantage Estimation，PPO 用）
# =====================================================================
@dataclass
class GAEConfig:
    gamma: float = 0.99
    lam: float = 0.95


def compute_gae(
    rewards: torch.Tensor,  # (T,)
    values: torch.Tensor,  # (T,)
    dones: torch.Tensor,  # (T,)
    cfg: GAEConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GAE 计算 advantage 和 returns。

    笔记对应截图里 PPO 那条路："Reward Model → r → GAE → A"
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + cfg.gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + cfg.gamma * cfg.lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# =====================================================================
# KL 散度（Policy vs Reference）
# =====================================================================
def kl_divergence(
    p_logits: torch.Tensor, q_logits: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """token-level KL(p || q)，按 mask 加权平均。

    笔记对应截图：PPO 和 GRPO 都有 Reference Model → KL。
    """
    p_log = F.log_softmax(p_logits, dim=-1)
    q_log = F.log_softmax(q_logits.detach(), dim=-1)
    p_prob = p_log.exp()
    kl = (p_prob * (p_log - q_log)).sum(dim=-1)  # (..., seq_len)
    if mask is None:
        return kl.mean()
    return (kl * mask).sum() / mask.sum().clamp(min=1.0)


# =====================================================================
# Policy + Value Model 接口
# =====================================================================
class PolicyWithValue(nn.Module):
    """带 Value Head 的 Policy 模型（PPO 用）。

    笔记对应截图 PPO 框图：
      "Policy Model → o → Value Model → v"
    """

    def __init__(self, base: nn.Module, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.base = base  # 共享的 backbone（MLLM）
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        hidden = self.base(inputs)  # (B, T, hidden_dim)
        logits = self.lm_head(hidden)  # (B, T, vocab_size)
        value = self.value_head(hidden).squeeze(-1)  # (B, T)
        return logits, value, hidden


class PolicyOnly(nn.Module):
    """不带 Value Head 的 Policy 模型（GRPO 用）。

    笔记对应截图 GRPO 框图：
      "Policy Model → o_1, o_2, ..., o_G → Group ... → A_1, A_2, ..., A_G"
      没有 Value Model。
    """

    def __init__(self, base: nn.Module, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.base = base
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        hidden = self.base(inputs)
        logits = self.lm_head(hidden)
        return logits, hidden


# =====================================================================
# PPO Loss（PPO 用）
# =====================================================================
@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    kl_coef: float = 0.1
    value_coef: float = 0.5
    gamma: float = 0.99
    lam: float = 0.95


class PPOLoss(nn.Module):
    """PPO clipped surrogate + value loss + KL penalty。

    笔记对应截图 PPO 框图（Policy Model → o, Value Model → v, Reference Model → KL, Reward Model → r, GAE → A）。
    """

    def __init__(self, cfg: PPOConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        new_logits: torch.Tensor,  # (B, T, V) — policy 当前的 logits
        old_logits: torch.Tensor,  # (B, T, V) — rollout 时 policy 的 logits（固定）
        ref_logits: torch.Tensor,  # (B, T, V) — Reference Model 的 logits（用于 KL）
        actions: torch.Tensor,  # (B, T)    — token ids
        advantages: torch.Tensor,  # (B, T)    — GAE 算出的 advantage
        returns: torch.Tensor,  # (B, T)    — value 回归目标
        values: torch.Tensor,  # (B, T)    — 当前的 value 预测
        mask: torch.Tensor,  # (B, T)    — 1 表示有效 token
    ) -> Tuple[torch.Tensor, dict]:
        B, T, V = new_logits.shape

        # policy ratio
        new_logp = F.log_softmax(new_logits, dim=-1)
        old_logp = F.log_softmax(old_logits.detach(), dim=-1)
        # 取 action 位置的对数概率
        new_logp_a = new_logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # (B, T)
        old_logp_a = old_logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        ratio = (new_logp_a - old_logp_a).exp()  # (B, T)

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * mask).sum() / mask.sum().clamp(min=1.0)

        # KL penalty（policy vs reference）
        kl = kl_divergence(new_logits, ref_logits, mask)

        # value loss
        value_loss = ((values - returns) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)

        total = policy_loss + self.cfg.value_coef * value_loss + self.cfg.kl_coef * kl

        return total, {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "kl": kl.item() if torch.is_tensor(kl) else kl,
        }


# =====================================================================
# GRPO Loss（GRPO 用）
# =====================================================================
@dataclass
class GRPOConfig:
    clip_eps: float = 0.2
    kl_coef: float = 0.1
    group_size: int = 4  # 同一 prompt 采样 G 次


class GRPOLoss(nn.Module):
    """GRPO：不用 Value Model，用 Group-relative advantage。

    笔记对应截图 GRPO 框图：
      "o_1, o_2, ..., o_G → r_1, r_2, ..., r_G → Group relative → A_1, A_2, ..., A_G"
      无 Value Model、无 GAE、无单独 Value head。

    优势函数：A_i = (r_i - mean(r_group)) / std(r_group)
    目标函数：和 PPO 一样的 clipped surrogate + KL penalty，但 advantage 是 group-relative。
    """

    def __init__(self, cfg: GRPOConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        new_logits: torch.Tensor,  # (B*G, T, V) — B 个 prompt，每个 G 个采样
        old_logits: torch.Tensor,  # 同上
        ref_logits: torch.Tensor,  # 同上
        actions: torch.Tensor,  # (B*G, T)
        rewards: torch.Tensor,  # (B*G,)    — 每条采样的总奖励（一个标量）
        mask: torch.Tensor,  # (B*G, T)
    ) -> Tuple[torch.Tensor, dict]:
        BG = new_logits.shape[0]
        G = self.cfg.group_size
        if G < 2 or BG % G:
            raise ValueError("Need complete groups with group_size >= 2")
        B = BG // G

        # === Group-relative advantage ===
        # rewards reshape 成 (B, G)，按组内归一化
        r_group = rewards.view(B, G)
        r_mean = r_group.mean(dim=1, keepdim=True)
        r_std = r_group.std(dim=1, keepdim=True).clamp(min=1e-4)
        advantages = ((r_group - r_mean) / r_std).view(-1)  # (B*G,)

        # broadcast 到 token 级别（GRPO 简化：每个 token 用同 prompt 的 advantage）
        advantages_t = advantages.unsqueeze(1).expand(
            -1, new_logits.shape[1]
        )  # (B*G, T)

        # === Clipped surrogate ===
        new_logp = F.log_softmax(new_logits, dim=-1)
        old_logp = F.log_softmax(old_logits.detach(), dim=-1)
        new_logp_a = new_logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        old_logp_a = old_logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        ratio = (new_logp_a - old_logp_a).exp()

        surr1 = ratio * advantages_t
        surr2 = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advantages_t
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * mask).sum() / mask.sum().clamp(min=1.0)

        # KL penalty
        kl = kl_divergence(new_logits, ref_logits, mask)

        total = policy_loss + self.cfg.kl_coef * kl

        return total, {
            "policy_loss": policy_loss.item(),
            "kl": kl.item() if torch.is_tensor(kl) else kl,
            "group_adv_std": r_std.mean().item(),
        }


# =====================================================================
# 对比图绘制（生成那张 PPO vs GRPO 图）
# =====================================================================
def plot_ppo_vs_grpo(save_path: str = "/tmp/ppo_vs_grpo.png") -> None:
    """绘制 PPO vs GRPO 架构对比图（对应笔记那张截图）。

    用 matplotlib 画结构图，不依赖 graphviz。
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib not installed, skip plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # === PPO 框图（上半部分）===
    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title(
        "PPO (4 models: Policy / Value / Reference / Reward)",
        fontsize=14,
        fontweight="bold",
    )

    # q box
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.2, 1.5), 0.6, 0.8, boxstyle="round,pad=0.05", fc="#FFF4D6", ec="black"
        )
    )
    ax.text(0.5, 1.9, "q", ha="center", va="center", fontsize=12)

    # Policy Model
    ax.add_patch(
        patches.FancyBboxPatch(
            (1.2, 1.5), 1.4, 0.8, boxstyle="round,pad=0.05", fc="#FFE08A", ec="black"
        )
    )
    ax.text(1.9, 1.9, "Policy\nModel", ha="center", va="center", fontsize=11)

    # o box
    ax.add_patch(
        patches.FancyBboxPatch(
            (3.0, 1.5), 0.6, 0.8, boxstyle="round,pad=0.05", fc="white", ec="black"
        )
    )
    ax.text(3.3, 1.9, "o", ha="center", va="center", fontsize=12)

    # Reference Model (Frozen)
    ax.add_patch(
        patches.FancyBboxPatch(
            (4.0, 2.5), 1.4, 0.8, boxstyle="round,pad=0.05", fc="#D6E4FF", ec="black"
        )
    )
    ax.text(4.7, 2.9, "Reference\nModel", ha="center", va="center", fontsize=11)

    # Reward Model (Frozen)
    ax.add_patch(
        patches.FancyBboxPatch(
            (4.0, 1.5), 1.4, 0.8, boxstyle="round,pad=0.05", fc="#D6E4FF", ec="black"
        )
    )
    ax.text(4.7, 1.9, "Reward\nModel", ha="center", va="center", fontsize=11)

    # Value Model (Trained)
    ax.add_patch(
        patches.FancyBboxPatch(
            (4.0, 0.5), 1.4, 0.8, boxstyle="round,pad=0.05", fc="#FFE08A", ec="black"
        )
    )
    ax.text(4.7, 0.9, "Value\nModel", ha="center", va="center", fontsize=11)

    # r, v boxes
    ax.add_patch(
        patches.FancyBboxPatch(
            (5.8, 1.7), 0.5, 0.4, boxstyle="round,pad=0.05", fc="white", ec="black"
        )
    )
    ax.text(6.05, 1.9, "r", ha="center", va="center", fontsize=12)
    ax.add_patch(
        patches.FancyBboxPatch(
            (5.8, 0.7), 0.5, 0.4, boxstyle="round,pad=0.05", fc="white", ec="black"
        )
    )
    ax.text(6.05, 0.9, "v", ha="center", va="center", fontsize=12)

    # GAE
    ax.add_patch(
        patches.FancyBboxPatch(
            (6.8, 1.0), 1.4, 0.8, boxstyle="round,pad=0.05", fc="white", ec="black"
        )
    )
    ax.text(7.5, 1.4, "GAE", ha="center", va="center", fontsize=12)

    # A
    ax.add_patch(
        patches.FancyBboxPatch(
            (8.6, 1.0), 0.5, 0.8, boxstyle="round,pad=0.05", fc="white", ec="black"
        )
    )
    ax.text(8.85, 1.4, "A", ha="center", va="center", fontsize=12)

    # 箭头
    ax.annotate("", xy=(1.2, 1.9), xytext=(0.8, 1.9), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(3.0, 1.9), xytext=(2.6, 1.9), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(4.0, 2.9), xytext=(3.6, 2.0), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(4.0, 1.9), xytext=(3.6, 1.9), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(4.0, 0.9), xytext=(3.6, 1.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(5.8, 1.9), xytext=(5.4, 1.9), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(5.8, 0.9), xytext=(5.4, 0.9), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(6.8, 1.4), xytext=(6.3, 1.85), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(6.8, 1.2), xytext=(6.3, 0.95), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(8.6, 1.4), xytext=(8.2, 1.4), arrowprops=dict(arrowstyle="->"))

    # KL 标注
    ax.text(4.7, 3.3, "KL", ha="center", fontsize=11, color="red")
    ax.plot([3.3, 4.0], [2.0, 2.9], "r--", lw=1)

    # 加号 ⊕ (拼接 r 和 v 进 GAE)
    ax.text(6.5, 1.4, "⊕", ha="center", va="center", fontsize=18)

    # 图例
    ax.add_patch(patches.Rectangle((10.5, 0.5), 0.4, 0.4, fc="#FFE08A", ec="black"))
    ax.text(11.0, 0.7, "Trained", va="center", fontsize=10)
    ax.add_patch(patches.Rectangle((10.5, 1.0), 0.4, 0.4, fc="#D6E4FF", ec="black"))
    ax.text(11.0, 1.2, "Frozen", va="center", fontsize=10)

    # === GRPO 框图（下半部分）===
    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title(
        "GRPO (3 models: Policy / Reference / Reward + Group baseline)",
        fontsize=14,
        fontweight="bold",
    )

    # Policy
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.5, 1.5), 1.4, 0.8, boxstyle="round,pad=0.05", fc="#FFE08A", ec="black"
        )
    )
    ax.text(1.2, 1.9, "Policy\nModel", ha="center", va="center", fontsize=11)

    # o_1, o_2, ..., o_G
    for i in range(4):
        ax.add_patch(
            patches.FancyBboxPatch(
                (2.3 + i * 0.6, 1.5),
                0.5,
                0.8,
                boxstyle="round,pad=0.05",
                fc="white",
                ec="black",
            )
        )
        ax.text(2.55 + i * 0.6, 1.9, f"o{i + 1}", ha="center", va="center", fontsize=10)

    # Reference Model
    ax.add_patch(
        patches.FancyBboxPatch(
            (5.0, 2.5), 1.4, 0.8, boxstyle="round,pad=0.05", fc="#D6E4FF", ec="black"
        )
    )
    ax.text(5.7, 2.9, "Reference\nModel", ha="center", va="center", fontsize=11)

    # Reward Model
    ax.add_patch(
        patches.FancyBboxPatch(
            (5.0, 0.5), 1.4, 0.8, boxstyle="round,pad=0.05", fc="#D6E4FF", ec="black"
        )
    )
    ax.text(5.7, 0.9, "Reward\nModel", ha="center", va="center", fontsize=11)

    # r_1..r_G
    for i in range(4):
        ax.add_patch(
            patches.FancyBboxPatch(
                (7.0 + i * 0.6, 0.5),
                0.5,
                0.8,
                boxstyle="round,pad=0.05",
                fc="white",
                ec="black",
            )
        )
        ax.text(7.25 + i * 0.6, 0.9, f"r{i + 1}", ha="center", va="center", fontsize=10)

    # Group relative
    ax.add_patch(
        patches.FancyBboxPatch(
            (7.0, 1.6), 2.0, 0.8, boxstyle="round,pad=0.05", fc="white", ec="black"
        )
    )
    ax.text(8.0, 2.0, "Group\nrelative", ha="center", va="center", fontsize=11)

    # A_1..A_G
    for i in range(4):
        ax.add_patch(
            patches.FancyBboxPatch(
                (9.6 + i * 0.6, 1.5),
                0.5,
                0.8,
                boxstyle="round,pad=0.05",
                fc="white",
                ec="black",
            )
        )
        ax.text(9.85 + i * 0.6, 1.9, f"A{i + 1}", ha="center", va="center", fontsize=10)

    # 箭头
    ax.annotate("", xy=(2.3, 1.9), xytext=(1.9, 1.9), arrowprops=dict(arrowstyle="->"))
    for i in range(4):
        ax.annotate(
            "",
            xy=(5.0, 2.9),
            xytext=(2.55 + i * 0.6, 2.0),
            arrowprops=dict(arrowstyle="->"),
        )
        ax.annotate(
            "",
            xy=(7.0 + i * 0.6, 0.9),
            xytext=(2.55 + i * 0.6, 1.6),
            arrowprops=dict(arrowstyle="->"),
        )
        ax.annotate(
            "",
            xy=(7.0 + i * 0.6, 1.9),
            xytext=(7.0 + i * 0.6, 1.4),
            arrowprops=dict(arrowstyle="->"),
        )
        ax.annotate(
            "",
            xy=(9.6 + i * 0.6, 1.9),
            xytext=(9.0, 2.0),
            arrowprops=dict(arrowstyle="->"),
        )

    # KL 标注
    ax.text(5.7, 3.3, "KL", ha="center", fontsize=11, color="red")
    ax.plot([3.5, 5.0], [2.5, 2.9], "r--", lw=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot] saved PPO vs GRPO diagram to {save_path}")


# =====================================================================
# Smoke test
# =====================================================================
if __name__ == "__main__":
    # 1) GAE
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.6, 0.7])
    dones = torch.tensor([0.0, 0.0, 1.0])
    adv, ret = compute_gae(rewards, values, dones, GAEConfig())
    print(f"[GAE] advantages={adv.tolist()}, returns={ret.tolist()}")

    # 2) PPO Loss
    B, T, V = 2, 8, 100
    ppo_loss = PPOLoss(PPOConfig())
    new_logits = torch.randn(B, T, V)
    old_logits = torch.randn(B, T, V)
    ref_logits = torch.randn(B, T, V)
    actions = torch.randint(0, V, (B, T))
    advantages = torch.randn(B, T)
    returns = torch.randn(B, T)
    values = torch.randn(B, T)
    mask = torch.ones(B, T)
    loss, comp = ppo_loss(
        new_logits, old_logits, ref_logits, actions, advantages, returns, values, mask
    )
    print(f"\n[PPO Loss] total={loss.item():.4f}, components={comp}")

    # 3) GRPO Loss
    B, G, T, V = 2, 4, 8, 100
    grpo_loss = GRPOLoss(GRPOConfig(group_size=G))
    new_logits = torch.randn(B * G, T, V)
    old_logits = torch.randn(B * G, T, V)
    ref_logits = torch.randn(B * G, T, V)
    actions = torch.randint(0, V, (B * G, T))
    rewards = torch.randn(B * G)
    mask = torch.ones(B * G, T)
    loss, comp = grpo_loss(new_logits, old_logits, ref_logits, actions, rewards, mask)
    print(f"\n[GRPO Loss] total={loss.item():.4f}, components={comp}")

    # 4) Plot
    plot_ppo_vs_grpo()
