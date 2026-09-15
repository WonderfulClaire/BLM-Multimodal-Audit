"""Bind review decisions to content and exclude held-out identities and media."""

from dataclasses import dataclass, field
import hashlib
import json


@dataclass
class HoldoutIndex:
    ids: set = field(default_factory=set)
    group_ids: set = field(default_factory=set)
    image_hashes: set = field(default_factory=set)
    case_hashes: set = field(default_factory=set)


def candidate_digest(candidate):
    content = {
        k: v
        for k, v in candidate.items()
        if k not in {"curation", "review", "selection"}
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def review_candidate(candidate, decision, holdout, rule_pass=True):
    digest = candidate_digest(candidate)

    def verdict(status, reason):
        return {
            "status": status,
            "reason": reason,
            "candidate_digest": digest,
            "reviewer": decision.get("reviewer"),
            "rule_version": candidate.get("rule_version"),
        }

    required = (
        "id",
        "source_id",
        "group_id",
        "split",
        "prompt",
        "ground_truth",
        "teacher_model",
        "rule_version",
    )
    required += (
        ("case", "case_sha256")
        if candidate.get("task_type") == "agent"
        else ("image", "image_sha256")
    )
    if any(not candidate.get(k) for k in required):
        return verdict("reject", "missing_provenance_or_fields")
    if (
        candidate["split"] != "train"
        or candidate["id"] in holdout.ids
        or candidate["source_id"] in holdout.ids
        or candidate["group_id"] in holdout.group_ids
        or candidate.get("image_sha256") in holdout.image_hashes
        or candidate.get("case_sha256") in holdout.case_hashes
    ):
        return verdict("reject", "evaluation_leakage")
    if candidate.get("task_type") == "agent":
        if (
            not isinstance(candidate["case"], dict)
            or not isinstance(candidate["ground_truth"], list)
            or not candidate["ground_truth"]
            or any(
                x not in [f"C{i}" for i in range(1, 9)]
                for x in candidate["ground_truth"]
            )
        ):
            return verdict("reject", "invalid_label_schema")
        if case_digest(candidate["case"]) != candidate["case_sha256"]:
            return verdict("reject", "case_content_mismatch")
    elif not isinstance(candidate["ground_truth"], dict) or any(
        not isinstance(candidate["ground_truth"].get(k), str)
        or not candidate["ground_truth"][k].strip()
        for k in ("risk_category", "risk_level", "reason")
    ):
        return verdict("reject", "invalid_label_schema")
    if (
        decision.get("candidate_id") != candidate["id"]
        or decision.get("candidate_digest") != digest
    ):
        return verdict("quarantine", "missing_or_stale_review")
    if (
        not decision.get("reviewer")
        or not decision.get("reason")
        or decision["reviewer"] == candidate["teacher_model"]
    ):
        return verdict("quarantine", "independent_review_required")
    if decision.get("decision") == "reject":
        return verdict("reject", "reviewer_rejected")
    if decision.get("decision") != "accept":
        return verdict("quarantine", "review_pending")
    if rule_pass is not True:
        return verdict("quarantine", "rule_judge_conflict")
    return verdict("accept", "passed")


def case_digest(case):
    return hashlib.sha256(
        json.dumps(case, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
