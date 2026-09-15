"""Train global, region and hard-negative alignment from a validated JSONL manifest."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .model import BLMCLIP
from .fg_clip import FGClipLoss
from .dynamic_resolution import SmartResizeConfig


def tokenize(text, length=64):
    # Deterministic byte tokenizer for the compact CPU model; no random text IDs.
    ids = [int(x) + 1 for x in text.encode("utf-8")[:length]]
    return torch.tensor(ids + [0] * (length - len(ids)))


def image_tensor(image, cfg):
    w, h = image.size
    h, w = cfg.smart_resize(h, w)
    arr = (
        np.asarray(image.convert("RGB").resize((w, h)), dtype=np.float32).copy() / 255.0
    )
    return torch.from_numpy(arr).permute(2, 0, 1)


def train(manifest, output, steps=3, seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rows = [json.loads(x) for x in Path(manifest).read_text().splitlines() if x.strip()]
    if len(rows) < 2:
        raise ValueError("Need at least two images for contrastive training")
    model = BLMCLIP(
        embed_dim=32,
        img_embed_dim=48,
        text_embed_dim=48,
        img_backbone_kwargs={"depth": 2, "num_heads": 6, "window_size": 4},
        text_backbone_kwargs={"num_layers": 1, "num_heads": 6},
    )
    criterion = FGClipLoss()
    optim = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), lr=1e-3
    )
    resize = SmartResizeConfig(min_pixels=28 * 28, max_pixels=112 * 112)
    examples = []
    for row in rows:
        image = Image.open(Path(manifest).parent / row["image"]).convert("RGB")
        examples.append((row, image, image_tensor(image, resize)))
    history = []
    for step in range(steps):
        # Variable-size images are encoded independently; token packing can schedule these groups.
        gi = torch.cat([model.encode_image(t.unsqueeze(0)) for _, _, t in examples])
        gt = model.encode_text(
            torch.stack([tokenize(row["caption"]) for row, _, _ in examples])
        )
        ri = []
        rt = []
        hn = []
        for row, image, _ in examples:
            for region in row["regions"]:
                ri.append(
                    model.encode_image(
                        image_tensor(image.crop(region["bbox"]), resize).unsqueeze(0)
                    )
                )
                rt.append(tokenize(region["positive"]))
                hn.append(tokenize(region["negatives"][0]))
        if not ri:
            raise ValueError("No verified region pairs")
        loss, parts = criterion(
            gi,
            gt,
            torch.cat(ri),
            model.encode_text(torch.stack(rt)),
            model.encode_text(torch.stack(hn)).unsqueeze(1),
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
        history.append({"step": step, "loss": loss.item(), **parts})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "loss": criterion.state_dict(), "seed": seed},
        output / "model.pt",
    )
    (output / "metrics.json").write_text(json.dumps(history, indent=2))
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--output", default="runs/visual")
    p.add_argument("--steps", type=int, default=3)
    a = p.parse_args()
    print(json.dumps(train(a.manifest, a.output, a.steps)))


if __name__ == "__main__":
    main()
