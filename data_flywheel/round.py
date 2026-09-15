"""Run a reproducible curation round from cached teacher outputs and reviews."""

import argparse
import hashlib
import json
from pathlib import Path
from .curation import HoldoutIndex, review_candidate, candidate_digest, case_digest
from .selection import select_failures
from .feedback import feedback_to_pairs
from .release import build_release
from .teacher import build_teacher_request


def read_rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def run_round(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    failures = read_rows(config["failures"])
    base = read_rows(config["base"])
    holdout = read_rows(config["holdout"])

    def materialize(rows, source_path):
        result = []
        for row in rows:
            row = dict(row)
            if row.get("task_type") == "agent":
                digest = case_digest(row["case"])
                if row.get("case_sha256") and row["case_sha256"] != digest:
                    raise ValueError("Case changed after annotation")
                row["case_sha256"] = digest
                result.append(row)
                continue
            image = (Path(source_path).parent / row["image"]).resolve()
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if row.get("image_sha256") and row["image_sha256"] != digest:
                raise ValueError("Image changed after annotation")
            row["image_sha256"] = digest
            row["image"] = str(image)
            result.append(row)
        return result

    base = materialize(base, config["base"])
    holdout = materialize(holdout, config["holdout"])
    index = HoldoutIndex(
        ids={r["id"] for r in holdout},
        group_ids={r["group_id"] for r in holdout},
        image_hashes={r["image_sha256"] for r in holdout if r.get("image_sha256")},
        case_hashes={r["case_sha256"] for r in holdout if r.get("case_sha256")},
    )
    if any(
        r["id"] in index.ids
        or r["group_id"] in index.group_ids
        or r.get("image_sha256") in index.image_hashes
        or r.get("case_sha256") in index.case_hashes
        for r in base
    ):
        raise ValueError("Base replay overlaps held-out data")
    selected = select_failures(
        failures, config.get("budget", 20), config.get("max_per_group", 2)
    )
    selected_ids = {r["id"] for r in selected}
    candidates = read_rows(config["candidates"])
    reviews = read_rows(config["reviews"])
    if len({r["candidate_id"] for r in reviews}) != len(reviews):
        raise ValueError("Duplicate or conflicting reviews require resolution")
    reviews = {r["candidate_id"]: r for r in reviews}
    rule_results = (
        {r["candidate_id"]: r for r in read_rows(config["rule_results"])}
        if config.get("rule_results")
        else {}
    )
    approved = []
    decisions = []
    for candidate in candidates:
        decision = review_candidate(
            candidate,
            reviews.get(candidate["id"], {}),
            index,
            rule_pass=(
                rule_results.get(candidate["id"], {}).get("passed") is True
                and rule_results[candidate["id"]].get("candidate_digest")
                == candidate_digest(candidate)
            ),
        )
        if candidate.get("task_type") != "agent":
            path = (Path(config["candidates"]).parent / candidate["image"]).resolve()
            if not path.is_file() or hashlib.sha256(
                path.read_bytes()
            ).hexdigest() != candidate.get("image_sha256"):
                decision.update(status="reject", reason="image_content_mismatch")
        if (
            candidate.get("source_id") not in selected_ids
            and decision["status"] != "reject"
        ):
            decision.update(status="quarantine", reason="source_not_selected")
        decisions.append({"id": candidate["id"], **decision})
        if decision["status"] == "accept":
            if (
                candidate.get("task_type") != "agent"
                and not Path(candidate["image"]).is_absolute()
            ):
                raise ValueError(
                    "Prepare candidate image as an absolute path before review binding"
                )
            approved.append({**candidate, "curation": decision})
    # Output is create-only. A failed round remains inspectable and cannot overwrite a prior release.
    output.mkdir(parents=True, exist_ok=False)

    def write(name, rows):
        (output / name).write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
        )

    write("selected.jsonl", selected)
    write(
        "teacher_requests.jsonl",
        [build_teacher_request(x, config["rules"]) for x in selected],
    )
    write("decisions.jsonl", decisions)
    pairs = []
    if config.get("feedback"):
        for feedback in read_rows(config["feedback"]):
            if feedback.get("source_id") in selected_ids:
                pairs.extend(feedback_to_pairs(feedback))
    write("preference_pairs.jsonl", pairs)
    report = build_release(
        base, approved, output / "dataset", config.get("max_new_fraction", 0.25)
    )
    report.update(
        selected=len(selected),
        approved=len(approved),
        preference_pairs=len(pairs),
        quarantined=sum(d["status"] == "quarantine" for d in decisions),
        rejected=sum(d["status"] == "reject" for d in decisions),
    )
    (output / "round_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run_round(json.loads(a.config.read_text()), a.output), indent=2))


if __name__ == "__main__":
    main()
