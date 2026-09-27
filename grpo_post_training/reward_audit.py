"""Audit whether a training reward agrees with an independent evaluation score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RewardAudit:
    n: int
    comparable_pairs: int
    inversions: int
    ties: int
    inversion_rate: float
    top_reward_failure_rate: float
    best_reward_index: int | None
    best_eval_index: int | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "n": self.n,
            "comparable_pairs": self.comparable_pairs,
            "inversions": self.inversions,
            "ties": self.ties,
            "inversion_rate": self.inversion_rate,
            "top_reward_failure_rate": self.top_reward_failure_rate,
            "best_reward_index": self.best_reward_index,
            "best_eval_index": self.best_eval_index,
        }


def audit_reward_alignment(
    rewards: Sequence[float],
    eval_scores: Sequence[float],
    *,
    success_threshold: float = 1.0,
) -> RewardAudit:
    """Compare reward ordering with an independent evaluation ordering."""
    if len(rewards) != len(eval_scores):
        raise ValueError("rewards and eval_scores must have the same length")
    if success_threshold < 0:
        raise ValueError("success_threshold must be nonnegative")
    n = len(rewards)
    if n == 0:
        return RewardAudit(0, 0, 0, 0, 0.0, 0.0, None, None)

    inversions = 0
    ties = 0
    comparable = 0
    for left in range(n):
        for right in range(left + 1, n):
            reward_delta = rewards[left] - rewards[right]
            eval_delta = eval_scores[left] - eval_scores[right]
            if reward_delta == 0 or eval_delta == 0:
                ties += 1
                continue
            comparable += 1
            if reward_delta * eval_delta < 0:
                inversions += 1

    best_reward = max(range(n), key=lambda i: rewards[i])
    best_eval = max(range(n), key=lambda i: eval_scores[i])
    max_reward = max(rewards)
    top_reward_indices = [i for i, value in enumerate(rewards) if value == max_reward]
    failures = sum(eval_scores[i] < success_threshold for i in top_reward_indices)

    return RewardAudit(
        n=n,
        comparable_pairs=comparable,
        inversions=inversions,
        ties=ties,
        inversion_rate=(inversions / comparable) if comparable else 0.0,
        top_reward_failure_rate=failures / len(top_reward_indices),
        best_reward_index=best_reward,
        best_eval_index=best_eval,
    )
