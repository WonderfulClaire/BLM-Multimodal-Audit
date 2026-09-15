"""Apply the flywheel release gate to actual Agent predictions and frozen cases."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .evaluation import compare_predictions


def compare_agent_predictions(cases, baseline, challenger):
    def index(rows, field):
        result = {r[field]: r for r in rows}
        if not rows or len(result) != len(rows):
            raise ValueError("Empty or duplicate evaluation IDs")
        return result

    def labels(value, expected=False):
        if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
            raise ValueError("Root causes must be a list of strings")
        if len(set(value)) != len(value):
            raise ValueError("Duplicate root causes")
        if expected and (not value or not set(value) <= {f"C{i}" for i in range(1, 9)}):
            raise ValueError("Missing or invalid independent labels")
        return tuple(sorted(value))

    truth = index(cases, "id")
    old, new = index(baseline, "scenario_id"), index(challenger, "scenario_id")
    if truth.keys() != old.keys() or truth.keys() != new.keys():
        raise ValueError("Predictions must exactly cover the evaluation manifest")
    if any(r.get("split") in (None, "train") for r in cases):
        raise ValueError("Training data or unspecified split cannot evaluate a release")
    paired = [[], []]
    strata = defaultdict(lambda: [[], []])
    failures = [0, 0]
    for key, case in truth.items():
        expected = labels(case["ground_truth"], expected=True)
        for side, rows in enumerate((old, new)):
            row = rows[key]
            if labels(row["ground_truth"], expected=True) != expected:
                raise ValueError("Prediction file changed the independent labels")
            prediction = labels(row["predicted_root_causes"])
            failure_count = row["tool_failure_count"]
            if type(failure_count) is not int or failure_count < 0:
                raise ValueError("Invalid tool failure count")
            failures[side] += failure_count
            paired[side].append({"id": key, "group_id": case.get("group_id", key),
                                 "expected": expected, "prediction": prediction})
            strata[str(len(expected))][side].append(int(prediction == expected))
    gate = compare_predictions(*paired)
    per_count = {key: {"baseline": sum(x)/len(x), "challenger": sum(y)/len(y),
                       "samples": len(x)} for key, (x, y) in strata.items()}
    retention = all(v["challenger"] >= v["baseline"] for v in per_count.values())
    protocol = failures[1] <= failures[0]
    independent_test = all(r["split"] == "test" for r in cases)
    statistical_pass = gate["accept"]
    accept = statistical_pass and retention and protocol and independent_test
    return {**gate, "statistical_gate_passed": statistical_pass,
            "by_root_cause_count": per_count, "no_cardinality_regression": retention,
            "tool_failures": {"baseline": failures[0], "challenger": failures[1]},
            "no_protocol_regression": protocol, "test_split_only": independent_test,
            "accept": accept, "decision": "candidate_for_review" if accept else "hold",
            "scope": "A test split label alone does not establish real-world validity or absence of prior tuning."}


def main():
    parser = argparse.ArgumentParser()
    for flag in ("cases", "baseline", "challenger", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = compare_agent_predictions(read(args.cases), read(args.baseline), read(args.challenger))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
