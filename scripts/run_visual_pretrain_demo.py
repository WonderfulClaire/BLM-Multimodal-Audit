"""Run the rendered-image data and three-loss training integration example."""

from pathlib import Path
import json
from scripts.make_fixture import generate
from visual_pretrain.pipeline import process_record
from visual_pretrain.train_manifest import train


def main():
    root = Path("runs/visual_demo")
    generate(root)
    rows = [
        process_record(json.loads(line), root)[0]
        for line in (root / "raw.jsonl").read_text().splitlines()
    ]
    manifest = root / "train.jsonl"
    manifest.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(json.dumps(train(manifest, root / "training")))


if __name__ == "__main__":
    main()
