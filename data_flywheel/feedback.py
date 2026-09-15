"""Convert explicit auditor corrections into preference pairs."""

import hashlib
import json


def feedback_to_pairs(feedback):
    required = (
        "id",
        "source_id",
        "image",
        "corrected_reason",
        "reviewer",
        "rule_version",
    )
    if any(not feedback.get(k) for k in required):
        raise ValueError("Incomplete auditor feedback")
    positive = feedback["corrected_reason"].strip()
    result = []
    seen = set()
    for suggestion in feedback.get("suggestions", []):
        negative = suggestion.get("text", "").strip()
        # Not chosen does not imply incorrect: this prevents false negatives.
        if (
            suggestion.get("verdict") != "incorrect"
            or not negative
            or negative == positive
            or negative in seen
        ):
            continue
        seen.add(negative)
        payload = {
            "source_id": feedback["source_id"],
            "image": feedback["image"],
            "chosen": positive,
            "rejected": negative,
            "reviewer": feedback["reviewer"],
            "rule_version": feedback["rule_version"],
            "feedback_id": feedback["id"],
            "training_use": "preference_or_region_hard_negative",
        }
        payload["id"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:20]
        result.append(payload)
    return result
