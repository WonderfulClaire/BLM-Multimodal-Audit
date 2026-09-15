"""Executable curation round with rendered media and explicit fixture reviews.

The renderer supplies known labels; no model-generated quality result is claimed.
"""

import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
from data_flywheel.curation import candidate_digest
from data_flywheel.round import run_round


def generate(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def image(name, color):
        path = root / name
        Image.new("RGB", (56, 56), color).save(path)
        return str(path), hashlib.sha256(path.read_bytes()).hexdigest()

    def row(i, color, split="train"):
        path, digest = image(f"{i}.png", color)
        return {
            "id": i,
            "source_id": i,
            "group_id": i,
            "split": split,
            "image": path,
            "image_sha256": digest,
            "prompt": "Synthetic rule: red is risk; all other colors are normal.",
            "ground_truth": {
                "risk_category": "risk" if color == "red" else "normal",
                "risk_level": "high" if color == "red" else "low",
                "reason": color + " square",
            },
            "teacher_model": "renderer-oracle",
            "rule_version": "color-v1",
        }

    base = [
        row("base" + str(i), color)
        for i, color in enumerate(["blue", "green", "yellow", "white"])
    ]
    holdout = [row("heldout", "black", "test")]
    failures = [
        {
            "id": "error1",
            "group_id": "source-red",
            "split": "train",
            "error_type": "false_negative",
            "severity": 1,
            "uncertainty": 0.8,
            "disagreement": 1,
            "cost": 1,
            "prediction": "normal",
        }
    ]
    c = row("aug1", "red")
    c.update(source_id="error1", group_id="source-red", error_type="false_negative")
    rejected = dict(
        c,
        id="leaked",
        image=holdout[0]["image"],
        image_sha256=holdout[0]["image_sha256"],
    )
    stale = dict(c, id="stale")
    stale_review = {
        "candidate_id": "stale",
        "candidate_digest": candidate_digest(stale),
        "decision": "accept",
        "reviewer": "fixture-reviewer",
        "reason": "Original label verified",
    }
    stale["ground_truth"] = {**stale["ground_truth"], "reason": "modified after review"}
    candidates = [c, rejected, stale]
    reviews = [
        {
            "candidate_id": x["id"],
            "candidate_digest": candidate_digest(x),
            "decision": "accept",
            "reviewer": "fixture-reviewer",
            "reason": "Renderer metadata reviewed",
        }
        for x in [c, rejected]
    ] + [stale_review]
    rule_results = [
        {
            "candidate_id": x["id"],
            "candidate_digest": candidate_digest(x),
            "passed": True,
            "validator": "renderer-fixture",
        }
        for x in candidates
    ]
    feedback = [
        {
            "id": "feedback1",
            "source_id": "error1",
            "image": c["image"],
            "corrected_reason": "red square",
            "reviewer": "fixture-human",
            "rule_version": "color-v1",
            "suggestions": [
                {"text": "blue square", "verdict": "incorrect"},
                {"text": "colored square", "verdict": "unselected"},
            ],
        }
    ]
    config = {
        "rules": {
            "version": "color-v1",
            "text": "Synthetic fixture rule: red means risk.",
        },
        "budget": 4,
        "max_new_fraction": 0.25,
    }
    for name, rows in [
        ("base", base),
        ("holdout", holdout),
        ("failures", failures),
        ("candidates", candidates),
        ("reviews", reviews),
        ("rule_results", rule_results),
        ("feedback", feedback),
    ]:
        path = root / (name + ".jsonl")
        path.write_text("".join(json.dumps(x) + "\n" for x in rows))
        config[name] = str(path)
    (root / "config.json").write_text(json.dumps(config, indent=2))
    return config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("runs/flywheel-demo"))
    a = p.parse_args()
    config = generate(a.output / "inputs")
    print(json.dumps(run_round(config, a.output / "round-001"), indent=2))


if __name__ == "__main__":
    main()
