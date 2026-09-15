"""Build verified over/under-diagnosis negatives for synthetic training only."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.run_agent_flywheel import validate_synthetic_case


def preference_case(row):
    if row.get("split") != "train":
        raise ValueError("Only training cases may produce preferences")
    chosen = sorted(row["ground_truth"])
    if chosen != validate_synthetic_case(row["case"]):
        raise ValueError("Synthetic rule verification failed")
    if len(chosen) == 1:
        negatives = [sorted(chosen + [c]) for c in (f"C{i}" for i in range(1, 9)) if c not in chosen]
        kind = "overdiagnosis"
    elif len(chosen) == 2:
        negatives = [[c] for c in chosen]
        kind = "underdiagnosis"
    else:
        raise ValueError("This fixture generator supports one or two causes")
    payload = json.dumps({"case": row["case"], "chosen": chosen, "negatives": negatives}, sort_keys=True)
    return {**row, "preference_candidates": negatives,
            "preference_audit": {"kind": kind, "validator": "synthetic-template-rules-v1",
                                 "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                                 "scope": "Rule-constructed candidates; not human preferences or sampled model errors"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [preference_case(json.loads(line)) for line in args.cases.read_text().splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    print(json.dumps({"cases": len(rows), "candidate_pairs": sum(len(r["preference_candidates"]) for r in rows)}))
