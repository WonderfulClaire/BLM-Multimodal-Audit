"""Task-specific teacher requests; suggestions are candidates, never auto-labels."""

import base64
import json
import mimetypes
from pathlib import Path
from urllib.request import Request, urlopen

ACTIONS = {
    "false_negative": "inspect_missed_evidence",
    "false_positive": "find_counterevidence",
    "wrong_reason": "correct_reason_with_visible_evidence",
    "unsupported_evidence": "replace_unsupported_claim",
    "tool_failure": "repair_tool_arguments",
    "missing_submission": "repair_terminal_action",
    "repeated_query": "remove_redundant_query",
    "low_reward": "diagnose_before_augmenting",
}


def build_teacher_request(failure, rules):
    action = ACTIONS.get(failure["error_type"])
    if not action:
        raise ValueError("Unsupported failure type")
    return {
        "source_id": failure["id"],
        "action": action,
        "rule_version": rules["version"],
        "rules": rules["text"],
        "observation": failure.get("observation"),
        "student_output": failure.get("prediction"),
        "requirements": [
            "Use only visible image evidence or returned tool observations.",
            "Separate facts from uncertainty.",
            "Return corrected_output, evidence, and explanation as JSON.",
            "If evidence is insufficient, set needs_human_review=true; do not invent labels.",
        ],
        "output_schema": {
            "corrected_output": "object",
            "evidence": "list",
            "explanation": "string",
            "needs_human_review": "boolean",
        },
    }


class TeacherClient:
    def __init__(self, base_url, model, api_key=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key

    def propose(self, request, image_path=None):
        content = [{"type": "text", "text": json.dumps(request, ensure_ascii=False)}]
        if image_path:
            path = Path(image_path)
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,"
                        + base64.b64encode(path.read_bytes()).decode()
                    },
                }
            )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        with urlopen(
            Request(self.url, data=json.dumps(payload).encode(), headers=headers),
            timeout=180,
        ) as response:
            result = json.load(response)
        return {
            "teacher_model": self.model,
            "source_id": request["source_id"],
            "rule_version": request["rule_version"],
            "proposal": json.loads(result["choices"][0]["message"]["content"]),
            "status": "needs_independent_review",
        }
