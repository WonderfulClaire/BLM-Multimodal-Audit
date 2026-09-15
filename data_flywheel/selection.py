"""Budgeted failure selection with explicit utility terms and source quotas."""

import math

ERROR_WEIGHTS = {
    "false_negative": 3.0,
    "false_positive": 2.0,
    "wrong_reason": 1.5,
    "tool_failure": 2.0,
    "unsupported_evidence": 2.5,
    "repeated_query": 1.0,
    "missing_submission": 2.0,
    "low_reward": 0.5,
}


def select_failures(rows, budget, max_per_group=2):
    if budget < 0 or max_per_group < 1:
        raise ValueError("Invalid selection budget")
    ranked = []
    for row in rows:
        if row.get("split") != "train" or row.get("error_type") not in ERROR_WEIGHTS:
            continue
        severity = float(row.get("severity", 0.5))
        uncertainty = float(row.get("uncertainty", 0))
        disagreement = float(row.get("disagreement", 0))
        cost = float(row.get("cost", 1))
        if (
            not all(
                math.isfinite(x) for x in (severity, uncertainty, disagreement, cost)
            )
            or cost <= 0
            or any(not 0 <= x <= 1 for x in (severity, uncertainty, disagreement))
        ):
            raise ValueError("Invalid utility inputs")
        utility = (
            ERROR_WEIGHTS[row["error_type"]] * (1 + severity)
            + uncertainty
            + disagreement
        )
        ranked.append(
            {
                **row,
                "selection": {
                    "utility": utility,
                    "estimated_cost": cost,
                    "utility_per_cost": utility / cost,
                },
            }
        )
    ranked.sort(key=lambda x: (-x["selection"]["utility_per_cost"], x["id"]))
    counts = {}
    used = 0
    selected = []
    seen = set()
    for row in ranked:
        group = row.get("group_id", row.get("source_id", row["id"]))
        cost = row["selection"]["estimated_cost"]
        if (
            row["id"] in seen
            or counts.get(group, 0) >= max_per_group
            or used + cost > budget
        ):
            continue
        selected.append(row)
        seen.add(row["id"])
        counts[group] = counts.get(group, 0) + 1
        used += cost
    return selected
