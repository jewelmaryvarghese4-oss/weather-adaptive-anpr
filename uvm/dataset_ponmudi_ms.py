"""
Like dataset_ponmudi.py, but the patches are taken at SEVERAL photo scales: for each sample the clear photo is treated as if it were W pixels wide,
W log-uniform from 1200 px up to the photo's own width (3000 to 4000 px for the real photos), and a 256 px patch of that version is cut out.
So the dehazer sees the same scene at the scale of a 1536 px copy and at the scale of the full-size photo. Patches never touch a detected plate.
"""
import csv
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from dataset_ponmudi import _boxes
from synth_ponmudi import ponmudi_degrade


class PonmudiMultiScaleDataset(Dataset):
    def __init__(self, photo_dir, list_csv, boxes_csv, patch_size=256, samples=3000, augment=True, seed=0, margin=0.3,
                 beta=(0.8, 2.4), min_width=1200):
        files = [r["file"] for r in csv.DictReader(open(list_csv, encoding="utf-8"))]
        bx = _boxes(boxes_csv)
        self.imgs, self.boxes = [], []
        for f in files:
            im = ImageOps.exif_transpose(Image.open(os.path.join(photo_dir, f))).convert("RGB")
            self.imgs.append(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
            self.boxes.append([((x1 - margin * (x2 - x1)), (y1 - margin * (y2 - y1)), (x2 + margin * (x2 - x1)), (y2 + margin * (y2 - y1)))
                               for (x1, y1, x2, y2) in bx.get(f, [])])
        self.patch, self.samples, self.augment, self.seed, self.beta, self.min_width = patch_size, samples, augment, seed, beta, min_width

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        rng = random.Random() if self.augment else random.Random(self.seed * 1_000_003 + idx)
        nprng = np.random.default_rng(None if self.augment else self.seed * 1_000_003 + idx)
        ps = self.patch
        for _ in range(80):
            k = rng.randrange(len(self.imgs))
            img = self.imgs[k]
            h, w = img.shape[:2]
            W = float(np.exp(rng.uniform(np.log(self.min_width), np.log(w))))     # pretend the photo is W px wide
            c = min(int(round(ps * w / W)), h, w)                                  # patch side in original pixels
            x0, y0 = rng.randint(0, w - c), rng.randint(0, h - c)
            if all(x0 + c <= a or x0 >= cc or y0 + c <= b or y0 >= d for a, b, cc, d in self.boxes[k]):
                break
        crop = cv2.resize(img[y0:y0 + c, x0:x0 + c], (ps, ps), interpolation=cv2.INTER_AREA)
        if self.augment and rng.random() < 0.5:
            crop = crop[:, ::-1].copy()
        bad = ponmudi_degrade(crop, nprng, beta=self.beta)
        t = lambda a: torch.from_numpy(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        return t(bad), t(crop)
