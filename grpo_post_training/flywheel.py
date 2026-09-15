"""Review teacher augmentations before adding them to the training set.

Teacher output is untrusted data. Approval is explicit, versioned, deduplicated,
and confined to train cases. Validation/test IDs cannot enter the loop.
"""

import argparse
import hashlib
import json
from pathlib import Path


def filter_candidates(candidates, decisions, forbidden_ids=()):
    approved = []
    rejected = []
    seen = set()
    decisions = {x["candidate_id"]: x for x in decisions}
    for candidate in candidates:
        cid = candidate["candidate_id"]
        decision = decisions.get(cid, {})
        reason = None
        if (
            candidate.get("split") != "train"
            or candidate.get("source_id") in forbidden_ids
        ):
            reason = "split_leakage"
        elif not candidate.get("teacher_model") or not candidate.get("rule_version"):
            reason = "missing_provenance"
        elif (
            not candidate.get("ground_truth")
            or not candidate.get("image")
            or not candidate.get("prompt")
        ):
            reason = "missing_training_fields"
        elif (
            decision.get("approved") is not True
            or not decision.get("reviewer")
            or not decision.get("reason")
        ):
            reason = "missing_approval"
        content = {k: candidate.get(k) for k in ("image", "prompt", "ground_truth")}
        digest = hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()
        if digest in seen:
            reason = "duplicate"
        if reason:
            rejected.append({"candidate_id": cid, "reason": reason})
            continue
        seen.add(digest)
        approved.append(
            {**candidate, "id": cid, "review": decision, "content_sha256": digest}
        )
    return approved, rejected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("candidates", type=Path)
    p.add_argument("decisions", type=Path)
    p.add_argument("--forbidden-ids", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    read = lambda path: [
        json.loads(x) for x in path.read_text().splitlines() if x.strip()
    ]
    approved, rejected = filter_candidates(
        read(a.candidates),
        read(a.decisions),
        set(json.loads(a.forbidden_ids.read_text())),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in approved)
    )
    a.output.with_suffix(".rejections.json").write_text(json.dumps(rejected, indent=2))
    print(json.dumps({"approved": len(approved), "rejected": len(rejected)}))


if __name__ == "__main__":
    main()
