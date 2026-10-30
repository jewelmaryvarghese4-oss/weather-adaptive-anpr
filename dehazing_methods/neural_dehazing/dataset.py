"""
dataset.py

Loads paired (hazy, clear) images produced by fog_synthesis.py.
Expects:
    root/clear/<name>.jpg
    root/hazy/<name>_v0.jpg, <name>_v1.jpg, ...

Each hazy variant is paired with its corresponding clear image.
Random crop + flip augmentation is applied identically to both images.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FogPairDataset(Dataset):
    def __init__(self, root: str, patch_size: int = 256, train: bool = True):
        self.root = Path(root)
        self.clear_dir = self.root / "clear"
        self.hazy_dir = self.root / "hazy"
        self.patch_size = patch_size
        self.train = train

        self.pairs = []
        for hazy_path in sorted(self.hazy_dir.iterdir()):
            # "<name>_v<k>.jpg" -> clear name is "<name>.jpg"
            stem = hazy_path.stem
            if "_v" not in stem:
                continue
            clear_name = stem.rsplit("_v", 1)[0] + hazy_path.suffix
            clear_path = self.clear_dir / clear_name
            if clear_path.exists():
                self.pairs.append((hazy_path, clear_path))

        if not self.pairs:
            raise RuntimeError(
                f"No hazy/clear pairs found under {root}. "
                "Did you run fog_synthesis.py first?"
            )

    def __len__(self):
        return len(self.pairs)

    def _load(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        hazy_path, clear_path = self.pairs[idx]
        hazy = self._load(hazy_path)
        clear = self._load(clear_path)

        h, w = clear.shape[:2]
        ps = self.patch_size

        # Resize up if the source image is smaller than the patch size
        if h < ps or w < ps:
            scale = ps / min(h, w)
            new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
            hazy = cv2.resize(hazy, (new_w, new_h))
            clear = cv2.resize(clear, (new_w, new_h))
            h, w = new_h, new_w

        if self.train:
            top = random.randint(0, h - ps)
            left = random.randint(0, w - ps)
        else:
            top = (h - ps) // 2
            left = (w - ps) // 2

        hazy = hazy[top:top + ps, left:left + ps]
        clear = clear[top:top + ps, left:left + ps]

        if self.train and random.random() < 0.5:
            hazy = np.fliplr(hazy).copy()
            clear = np.fliplr(clear).copy()

        hazy_t = torch.from_numpy(hazy).permute(2, 0, 1).float() / 255.0
        clear_t = torch.from_numpy(clear).permute(2, 0, 1).float() / 255.0

        return hazy_t, clear_t
