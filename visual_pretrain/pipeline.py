"""Caption -> phrases -> grounding -> verified hard negatives -> JSONL.

Cached mode consumes reviewed model outputs. Model mode runs a local Qwen2.5-VL,
spaCy and YOLO-World. Nothing silently substitutes random captions or boxes.
"""

import argparse
import json
import hashlib
from pathlib import Path
from PIL import Image
from .fg_clip import BBoxGenerator, HardNegativeBuilder


class QwenCaptioner:
    def __init__(self, model_path):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        ).eval()

    def __call__(self, image):
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "Describe visible objects, colors, actions and spatial relations accurately in English. Do not infer hidden facts.",
                    },
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.processor.batch_decode(
            ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]


class SpacyExtractor:
    def __init__(self, model_name="en_core_web_sm"):
        import spacy

        self.nlp = spacy.load(model_name)

    def __call__(self, caption):
        doc = self.nlp(caption)
        try:
            phrases = [c.text for c in doc.noun_chunks]
        except NotImplementedError:
            phrases = [
                "".join(t.text_with_ws for t in token.subtree).strip()
                for token in doc
                if token.pos_ in {"NOUN", "PROPN"}
            ]
        return list(dict.fromkeys(p for p in phrases if p.strip()))


class YoloGrounder:
    def __init__(self, weights):
        from ultralytics import YOLOWorld

        self.model = YOLOWorld(weights)

    def __call__(self, image, phrases):
        if not phrases:
            return []
        self.model.set_classes(phrases)
        result = self.model.predict(image, verbose=False)[0]
        return [
            {"phrase": phrases[int(label)], "bbox": box, "score": score}
            for box, score, label in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
            )
        ]


def process_record(
    record, root, captioner=None, extractor=None, grounder=None, verifier=None
):
    image_path = (Path(root) / record["image"]).resolve()
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    caption = record.get("caption") or (captioner(image) if captioner else None)
    if not caption:
        raise ValueError("Missing caption; supply cache or caption model")
    phrases = record.get("phrases") or (extractor(caption) if extractor else None)
    if not phrases:
        raise ValueError("Missing phrases; supply cache or phrase extractor")
    detections = record.get("detections")
    if detections is None:
        if grounder is None:
            raise ValueError("Missing boxes; supply cache or detector")
        detections = grounder(image, phrases)
    valid = []
    rejected = []
    for d in detections:
        box = d["bbox"]
        phrase = d["phrase"]
        score = float(d["score"])
        if (
            len(box) != 4
            or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)
            or phrase not in phrases
            or not 0 <= score <= 1
        ):
            rejected.append({"phrase": phrase, "reason": "invalid_detection"})
            continue
        valid.append(d)
    kept = []
    for phrase in phrases:
        subset = [d for d in valid if d["phrase"] == phrase]
        ids = BBoxGenerator().nms(
            [d["bbox"] for d in subset], [d["score"] for d in subset]
        )
        kept.extend(subset[i] for i in ids)
    regions = []
    for d in kept:
        phrase = d["phrase"]
        entry = record.get("negatives", {}).get(phrase)
        if entry:
            negative = entry["text"]
            approved = entry.get("verified") is True
        else:
            try:
                negative = HardNegativeBuilder().build(phrase)
            except ValueError:
                rejected.append({"phrase": phrase, "reason": "no_negative_candidate"})
                continue
            approved = bool(verifier and verifier(image, d["bbox"], phrase, negative))
        if not approved or negative.strip() == phrase.strip():
            rejected.append({"phrase": phrase, "reason": "negative_needs_verification"})
            continue
        regions.append(
            {
                "bbox": d["bbox"],
                "positive": phrase,
                "negatives": [negative],
                "score": d["score"],
            }
        )
    result = {
        "id": record["id"],
        "image": record["image"],
        "caption": caption,
        "regions": regions,
        "provenance": {
            "source": record.get("source", "user_supplied"),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "negative_verification": "explicit_review_or_verifier",
        },
    }
    return result, rejected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--caption-model")
    p.add_argument("--spacy-model")
    p.add_argument("--yolo-weights")
    a = p.parse_args()
    captioner = QwenCaptioner(a.caption_model) if a.caption_model else None
    extractor = SpacyExtractor(a.spacy_model) if a.spacy_model else None
    grounder = YoloGrounder(a.yolo_weights) if a.yolo_weights else None
    output = []
    rejects = []
    for line in a.input.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result, bad = process_record(
            row, a.input.parent, captioner, extractor, grounder
        )
        # Output image paths remain valid when the manifest is written elsewhere.
        import os

        result["image"] = os.path.relpath(
            a.input.parent / row["image"], a.output.parent
        )
        output.append(result)
        rejects.extend({"id": row["id"], **x} for x in bad)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in output)
    )
    a.output.with_suffix(".rejections.json").write_text(
        json.dumps(rejects, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                "images": len(output),
                "regions": sum(len(x["regions"]) for x in output),
                "rejected": len(rejects),
            }
        )
    )


if __name__ == "__main__":
    main()
