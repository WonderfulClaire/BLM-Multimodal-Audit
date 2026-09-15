"""Image records from reviewed manifests, without synthetic fallback."""

import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, manifest):
        self.path = Path(manifest)
        self.rows = [
            json.loads(line)
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return row, Image.open(self.path.parent / row["image"]).convert("RGB")
