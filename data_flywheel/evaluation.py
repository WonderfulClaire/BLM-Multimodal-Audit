"""Paired, source-group-bootstrap release gate on independent correctness labels."""

import random
from collections import defaultdict


def compare_predictions(
    baseline,
    challenger,
    min_delta=0.01,
    max_category_drop=0.02,
    bootstrap=1000,
    seed=42,
):
    def index(rows):
        result = {r["id"]: r for r in rows}
        if len(result) != len(rows) or not result:
            raise ValueError("Missing/duplicate evaluation IDs")
        return result

    b = index(baseline)
    c = index(challenger)
    if b.keys() != c.keys():
        raise ValueError("Evaluation IDs must be paired")
    groups = defaultdict(list)
    categories = defaultdict(list)
    delta = []
    for key in sorted(b):
        x, y = b[key], c[key]
        if x["expected"] != y["expected"] or x.get("group_id", key) != y.get(
            "group_id", key
        ):
            raise ValueError("Changed labels or evaluation groups")
        d = int(y["prediction"] == y["expected"]) - int(
            x["prediction"] == x["expected"]
        )
        delta.append(d)
        groups[x.get("group_id", key)].append(d)
        categories[str(x["expected"])].append(d)
    rng = random.Random(seed)
    names = sorted(groups)
    samples = []
    for _ in range(bootstrap):
        values = [v for name in rng.choices(names, k=len(names)) for v in groups[name]]
        samples.append(sum(values) / len(values))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    change = sum(delta) / len(delta)
    per = {k: sum(v) / len(v) for k, v in categories.items()}
    accept = (
        change >= min_delta
        and lo > 0
        and all(v >= -max_category_drop for v in per.values())
    )
    return {
        "samples": len(delta),
        "source_groups": len(groups),
        "accuracy_delta": change,
        "delta_ci95": [lo, hi],
        "category_deltas": per,
        "accept": accept,
        "decision": "candidate_for_review" if accept else "hold",
        "seed": seed,
        "bootstrap": bootstrap,
        "metric": "exact_correctness; reward not used for acceptance",
    }
