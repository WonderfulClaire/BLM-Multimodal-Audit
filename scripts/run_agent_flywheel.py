"""Curate training-only synthetic RCA corrections from actual model traces.

The teacher is an explicit synthetic-case composer and labels are checked by an
independent rule function for these templates. This is not a human/LLM Judge run.
"""

import argparse
import json
import re
from pathlib import Path
from data_flywheel.rl_feedback import groups_from_agent_traces
from data_flywheel.curation import case_digest, candidate_digest
from data_flywheel.round import run_round, read_rows


def validate_synthetic_case(case):
    s = case["sections"]
    causes = []

    def text(view):
        return " ".join(map(str, s.get(view, [])))

    speed = re.search(r"vehicle travels at (\d+) km/h", text("mobility"))
    if speed and int(speed[1]) > 40:
        causes.append("C1")
    if "excessive downtilt" in text("antenna"):
        causes.append("C2")
    distance = re.search(r"separation is ([\d.]+) km", text("cell_relation"))
    if distance and float(distance[1]) > 1:
        causes.append("C3")
    if "same frequency" in text("cell_relation") and "different site" in text(
        "cell_relation"
    ):
        causes.append("C4")
    if "identical remainders modulo 30" in text("cell_relation"):
        causes.append("C5")
    if "(ping-pong)" in text("handover"):
        causes.append("C6")
    if "incorrect threshold" in text("handover"):
        causes.append("C7")
    rb = re.search(r"show (\d+) resource blocks", text("resource"))
    if rb and int(rb[1]) < 160:
        causes.append("C8")
    return causes


def prepare(inputs, output, validator_name="template"):
    if validator_name == "measured":
        from data_flywheel.numeric_audit import validate_measured_case, RULE_VERSION
        validate = validate_measured_case
        rule_version = RULE_VERSION
    elif validator_name == "template":
        validate = validate_synthetic_case
        rule_version = "synthetic-rca-v1"
    else:
        raise ValueError("Unknown validator")
    inputs = Path(inputs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    routes = groups_from_agent_traces(read_rows(inputs / "trajectories.jsonl"))
    route_by_case = {r["source_id"]: r for r in routes}

    def enrich(row):
        return {
            **row,
            "task_type": "agent",
            "group_id": row["id"],
            "prompt": row["symptom"],
            "case_sha256": case_digest(row["case"]),
        }

    base = [enrich(x) for x in read_rows(inputs / "train.jsonl")]
    holdout = [enrich(x) for x in read_rows(inputs / "eval12.jsonl")]
    failures = []
    candidates = []
    reviews = []
    checks = []
    for row in read_rows(inputs / "compound_train.jsonl"):
        route = route_by_case.get(row["id"])
        if not route or route["route"] not in {"rl_ready", "teacher_or_sft_repair"}:
            continue
        if row["split"] != "train":
            raise ValueError("Refuse evaluation-source augmentation")
        failures.append(
            {
                "id": row["id"],
                "source_id": row["id"],
                "group_id": row["id"],
                "split": "train",
                "error_type": "wrong_reason",
                "severity": 1 - route["mean_correctness"],
                "uncertainty": min(1, route["correctness_std"]),
                "cost": 1,
                "observation": row["case"],
                "learning_route": route["route"],
            }
        )
        c = {
            **enrich(row),
            "id": "reviewed-" + row["id"],
            "source_id": row["id"],
            "teacher_model": "synthetic-case-composer-v1",
            "rule_version": rule_version,
            "error_type": "partial_diagnosis",
        }
        try:
            valid = validate(c["case"]) == sorted(c["ground_truth"])
        except (ValueError, KeyError, TypeError):
            valid = False
        digest = candidate_digest(c)
        candidates.append(c)
        reviews.append(
            {
                "candidate_id": c["id"],
                "candidate_digest": digest,
                "reviewer": "synthetic-rule-validator-v1",
                "decision": "accept" if valid else "reject",
                "reason": "Root-cause labels checked against explicit template measurements; synthetic scope only",
            }
        )
        checks.append(
            {
                "candidate_id": c["id"],
                "candidate_digest": digest,
                "passed": valid,
                "validator": rule_version,
            }
        )
    config = {
        "budget": 8,
        "max_new_fraction": 0.25,
        "rules": {
            "version": rule_version,
            "text": "Use only observed synthetic measurements. Include every supported cause.",
        },
    }
    for key, rows in [
        ("base", base),
        ("holdout", holdout),
        ("failures", failures),
        ("candidates", candidates),
        ("reviews", reviews),
        ("rule_results", checks),
    ]:
        path = (output / (key + ".jsonl")).resolve()
        path.write_text("".join(json.dumps(x) + "\n" for x in rows))
        config[key] = str(path)
    (output / "routes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in routes))
    (output / "config.json").write_text(json.dumps(config, indent=2))
    return config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validator", choices=["template", "measured"], default="template")
    a = p.parse_args()
    config = prepare(a.inputs, a.output / "inputs", validator_name=a.validator)
    print(json.dumps(run_round(config, a.output / "round-001"), indent=2))


if __name__ == "__main__":
    main()
