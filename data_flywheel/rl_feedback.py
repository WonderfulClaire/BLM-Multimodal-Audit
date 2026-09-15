"""Route sampled groups by learning signal, not by reward magnitude alone."""

import math
from statistics import pstdev


def route_group(rewards, correctness_scores, epsilon=1e-6):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if len(rewards) < 2 or len(rewards) != len(correctness_scores):
        raise ValueError("Need aligned rollout groups")
    if any(not math.isfinite(float(x)) for x in [*rewards, *correctness_scores]):
        raise ValueError("Nonfinite rollout scores")
    if any(not 0 <= x <= 1 for x in correctness_scores):
        raise ValueError("Correctness must be between 0 and 1")
    reward_std = pstdev(rewards)
    quality_std = pstdev(correctness_scores)
    inversions = sum(
        abs(correctness_scores[i] - correctness_scores[j]) > epsilon
        and abs(rewards[i] - rewards[j]) > epsilon
        and (correctness_scores[i] - correctness_scores[j]) * (rewards[i] - rewards[j]) < 0
        for i in range(len(rewards)) for j in range(i)
    )
    if all(x == 1 for x in correctness_scores):
        route = "efficiency_rl" if reward_std > epsilon else "mastered_replay"
    elif quality_std > epsilon and reward_std > epsilon:
        route = "audit_reward_quality_conflict" if inversions else "rl_ready"
    elif quality_std > epsilon:
        route = "repair_reward_resolution"
    elif reward_std > epsilon:
        route = "audit_reward_only_variance"
    else:
        route = "teacher_or_sft_repair"
    return {
        "route": route,
        "reward_std": reward_std,
        "correctness_std": quality_std,
        "mean_correctness": sum(correctness_scores) / len(correctness_scores),
        "group_size": len(rewards),
        "reward_quality_inversions": inversions,
    }


def groups_from_agent_traces(rows):
    groups = {}
    for row in rows:
        if row.get("split") != "train":
            continue
        key = (row["step"], row["case_id"])
        groups.setdefault(key, []).append(row)
    result = []
    for (step, case_id), members in sorted(groups.items()):
        scores = []
        for member in members:
            finals = [
                c
                for t in member["trace"]
                for c in t.get("info", {}).get("tool_calls", [])
                if "predicted_root_causes" in c
            ]
            if finals:
                predicted = set(finals[-1]["predicted_root_causes"])
                expected = set(finals[-1].get("ground_truth", []))
                if not expected:
                    raise ValueError("Missing independent environment labels")
                scores.append(
                    2 * len(predicted & expected) / (len(predicted) + len(expected))
                )
            else:
                scores.append(0.0)
        if len(members) < 2:
            continue
        result.append(
            {
                "id": f"{case_id}-round-{step}",
                "source_id": case_id,
                "step": step,
                "split": "train",
                **route_group([m["reward"] for m in members], scores),
            }
        )
    return result


def main():
    import argparse, json
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("trajectories", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(x) for x in a.trajectories.read_text().splitlines() if x.strip()]
    groups = groups_from_agent_traces(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(x) + "\n" for x in groups))
    counts = {}
    for group in groups:
        counts[group["route"]] = counts.get(group["route"], 0) + 1
    print(json.dumps({"groups": len(groups), "routes": counts}, indent=2))


if __name__ == "__main__":
    main()
