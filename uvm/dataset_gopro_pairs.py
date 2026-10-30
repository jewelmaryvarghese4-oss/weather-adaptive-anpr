"""
GoProHazy matched frame pairs (real fog / clear road, same place, NOT pixel-aligned) for training without a pixel loss against the clear frame.
Each item: a random scale of both frames (same resize), the same crop window in both, the same horizontal flip. Because the clear pass is a
few metres off the hazy pass, the two crops show the same kind of scene (sky on top, road below, same vanishing point) but not the same pixels.
"""
import csv
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class GoProPairs(Dataset):
    def __init__(self, root, split="train", crop=320, samples=4000, scale_range=(0.5, 1.0), augment=True, seed=0):
        self.root, self.crop, self.samples, self.sr, self.augment, self.seed = root, crop, samples, scale_range, augment, seed
        self.rows = [r for r in csv.DictReader(open(os.path.join(root, "pairs", "pairs.csv"), encoding="utf-8")) if r["split"] == split]
        self.split = split

    def __len__(self):
        return self.samples if self.augment else len(self.rows)

    def __getitem__(self, idx):
        rng = random.Random() if self.augment else random.Random(self.seed * 1_000_003 + idx)
        r = self.rows[rng.randrange(len(self.rows)) if self.augment else idx]
        d = os.path.join(self.root, "pairs", self.split)
        h, c = cv2.imread(os.path.join(d, r["hazy_file"])), cv2.imread(os.path.join(d, r["clear_file"]))
        s = rng.uniform(*self.sr) if self.augment else 0.5
        h, c = (cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for x in (h, c))
        H, W = h.shape[:2]
        S = self.crop
        y0, x0 = rng.randint(0, H - S), rng.randint(0, W - S)
        h, c = h[y0:y0 + S, x0:x0 + S], c[y0:y0 + S, x0:x0 + S]
        if self.augment and rng.random() < 0.5:
            h, c = h[:, ::-1], c[:, ::-1]
        t = lambda x: torch.from_numpy(cv2.cvtColor(np.ascontiguousarray(x), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        return t(h), t(c)
