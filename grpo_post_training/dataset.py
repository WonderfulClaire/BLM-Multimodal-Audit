"""Train-only audit manifests and explicit evaluation split exclusion."""

import json
from pathlib import Path


def load_training_rows(path, forbidden_ids=()):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError("Empty training manifest")
    seen = set()
    for row in rows:
        if row.get("split") != "train" or row["id"] in forbidden_ids:
            raise ValueError("Evaluation data cannot enter training")
        if row["id"] in seen:
            raise ValueError("Duplicate training ID")
        seen.add(row["id"])
    return rows
