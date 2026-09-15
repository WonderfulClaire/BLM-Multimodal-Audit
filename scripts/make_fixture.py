"""Render license-free colored-shape fixtures with exact boxes; no business data."""

import json, sys
from pathlib import Path
from PIL import Image, ImageDraw


def generate(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    audit = []
    for i, (color, cn, other) in enumerate(
        [("red", "红色", "蓝色"), ("blue", "蓝色", "红色"), ("green", "绿色", "黄色")]
    ):
        image = Image.new("RGB", (112 + 28 * i, 112), "white")
        draw = ImageDraw.Draw(image)
        box = [28, 28, 84, 84]
        draw.rectangle(box, fill=color)
        name = f"shape_{i}.png"
        image.save(root / name)
        phrase = cn + "方块"
        rows.append(
            {
                "id": f"synthetic-{i}",
                "image": name,
                "source": "synthetic_renderer",
                "caption": "白色背景中有一个" + phrase,
                "phrases": [phrase],
                "detections": [{"phrase": phrase, "bbox": box, "score": 1.0}],
                "negatives": {phrase: {"text": other + "方块", "verified": True}},
            }
        )
        # Artificial color-audit rule, explicitly unrelated to real moderation policy.
        audit.append(
            {
                "id": f"synthetic-{i}",
                "image": name,
                "prompt": "测试规则：红色方块标为风险，其他颜色标为正常。",
                "stage": "reasoning" if i < 2 else "general",
                "split": "train",
                "rule_version": "synthetic-color-v1",
                "ground_truth": {
                    "risk_category": "风险" if i == 0 else "正常",
                    "risk_level": "高危" if i == 0 else "低危",
                    "reason": cn + "方块",
                },
            }
        )
    for name, data in [("raw.jsonl", rows), ("audit.jsonl", audit)]:
        (root / name).write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data)
        )


if __name__ == "__main__":
    generate(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/fixtures"))
