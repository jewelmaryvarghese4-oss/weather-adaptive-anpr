"""
Training pairs made on the fly: a clear photo (or video frame) and the same picture turned into "rainy window in fog" with
synth_rainfog.degrade(). Returns (degraded, clear) tensors like the other dehaze datasets, so it can be concatenated with them.

Clear sources are folders of images; each epoch takes `samples` random crops (random image, random scale, random place), so
the same frame gives different fog / drops / noise every time. With augment=False the crops and the degradation are fixed per
index (a stable validation set).
"""
import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from synth_rainfog import degrade


class SynthRainFogDataset(Dataset):
    def __init__(self, clear_roots, patch_size=256, samples=3000, augment=True, seed=0, exclude=(), only=()):
        self.files = []
        for root in clear_roots:
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
                self.files += glob.glob(os.path.join(root, "**", ext), recursive=True)
        self.files = sorted(set(self.files))
        if exclude:
            self.files = [f for f in self.files if not any(e in f for e in exclude)]
        if only:
            self.files = [f for f in self.files if any(o in f for o in only)]
        if not self.files:
            raise RuntimeError(f"no clear images found under {clear_roots}")
        self.patch = patch_size
        self.samples = samples
        self.augment = augment
        self.seed = seed

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        rng = random.Random() if self.augment else random.Random(self.seed * 1_000_003 + idx)
        nprng = np.random.default_rng(None if self.augment else self.seed * 1_000_003 + idx)
        img = cv2.imread(rng.choice(self.files), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        # random scale: plates in the real photos are 150-250 px wide, in video frames 40-150 px
        s = rng.uniform(0.5, 1.0)
        ps = self.patch
        cw, ch = min(w, int(ps / s)), min(h, int(ps / s))
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        crop = img[y0:y0 + ch, x0:x0 + cw]
        crop = cv2.resize(crop, (ps, ps), interpolation=cv2.INTER_AREA)
        if self.augment and rng.random() < 0.5:
            crop = crop[:, ::-1].copy()
        bad = degrade(crop, nprng)
        to_t = lambda a: torch.from_numpy(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        return to_t(bad), to_t(crop)
