"""Immutable dataset snapshots retaining replay examples and curation lineage."""

import hashlib
import json
import math
from pathlib import Path
from .curation import candidate_digest


def build_release(base, approved, output, max_new_fraction=0.25):
    if not 0 < max_new_fraction < 1:
        raise ValueError("Replay fraction must be positive")
    if not base:
        raise ValueError("Base replay data is required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = list(base)
    ids = set()
    media = set()
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Replay contains evaluation data")
        if row["id"] in ids:
            raise ValueError("Duplicate base ID")
        ids.add(row["id"])
        media.add((row.get("image_sha256", row.get("case_sha256")), row.get("prompt")))
    limit = math.floor(len(base) * max_new_fraction / (1 - max_new_fraction))
    included = []
    skipped = []
    for row in approved:
        c = row.get("curation", {})
        key = (row.get("image_sha256", row.get("case_sha256")), row.get("prompt"))
        reason = None
        if c.get("status") != "accept" or c.get("candidate_digest") != candidate_digest(
            row
        ):
            reason = "not_approved_or_changed"
        elif row["id"] in ids or key in media:
            reason = "duplicate"
        elif len(included) >= limit:
            reason = "new_data_quota"
        if reason:
            skipped.append({"id": row.get("id"), "reason": reason})
            continue
        included.append(row)
        ids.add(row["id"])
        media.add(key)
    rows += included
    text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    report = {
        "schema_version": 1,
        "base_count": len(base),
        "new_count": len(included),
        "new_fraction": len(included) / len(rows),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "sample_ids": [x["id"] for x in rows],
        "skipped": skipped,
    }
    # Atomic directory creation prevents accidental overwrite by a second process.
    output.mkdir(parents=True, exist_ok=False)
    (output / "train.jsonl").write_text(text)
    (output / "manifest.json").write_text(json.dumps(report, indent=2))
    return report
