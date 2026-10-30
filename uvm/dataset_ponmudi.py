"""
Training pairs for the dehazer from the CLEAR Ponmudi photos: (fogged crop, clear crop), fog tuned to the real foggy photos (synth_ponmudi.py).

Fairness: the clear photos show cars whose plates are in our foggy TEST set. A patch is skipped if it overlaps (even a little) any detected plate
box of any car, so the dehazer never sees those plates, only the rest of the scene (ground, bodies, trees, grass).
"""
import ast
import csv
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from synth_ponmudi import ponmudi_degrade


def _boxes(boxes_csv):
    out = {}
    for r in csv.DictReader(open(boxes_csv, encoding="utf-8")):
        out.setdefault(r["file"], []).append((float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])))
    return out


class PonmudiFogDataset(Dataset):
    def __init__(self, photo_dir, list_csv, boxes_csv, patch_size=256, samples=3000, augment=True, seed=0, margin=0.3, beta=(0.8, 2.4)):
        files = [r["file"] for r in csv.DictReader(open(list_csv, encoding="utf-8"))]
        self.imgs, self.boxes = [], []
        bx = _boxes(boxes_csv)
        for f in files:
            im = ImageOps.exif_transpose(Image.open(os.path.join(photo_dir, f))).convert("RGB")
            s = 2000 / max(im.size)                                   # work on a 2000 px copy: patches then have about the plate scale of the real photos
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            self.imgs.append(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
            bb = []
            for (x1, y1, x2, y2) in bx.get(f, []):
                w, h = x2 - x1, y2 - y1
                bb.append(((x1 - margin * w) * s, (y1 - margin * h) * s, (x2 + margin * w) * s, (y2 + margin * h) * s))
            self.boxes.append(bb)
        self.patch, self.samples, self.augment, self.seed, self.beta = patch_size, samples, augment, seed, beta

    def __len__(self):
        return self.samples

    def _free(self, bb, x0, y0, ps):
        return all(x0 + ps <= a or x0 >= c or y0 + ps <= b or y0 >= d for a, b, c, d in bb)

    def __getitem__(self, idx):
        rng = random.Random() if self.augment else random.Random(self.seed * 1_000_003 + idx)
        nprng = np.random.default_rng(None if self.augment else self.seed * 1_000_003 + idx)
        ps = self.patch
        for _ in range(60):
            k = rng.randrange(len(self.imgs))
            img = self.imgs[k]
            h, w = img.shape[:2]
            sc = rng.uniform(0.6, 1.0)                                # crop size in the 2000 px image, resized to ps
            c = int(ps / sc)
            c = min(c, h, w)
            x0, y0 = rng.randint(0, w - c), rng.randint(0, h - c)
            if self._free(self.boxes[k], x0, y0, c):
                break
        crop = cv2.resize(img[y0:y0 + c, x0:x0 + c], (ps, ps), interpolation=cv2.INTER_AREA)
        if self.augment and rng.random() < 0.5:
            crop = crop[:, ::-1].copy()
        bad = ponmudi_degrade(crop, nprng, beta=self.beta)
        t = lambda a: torch.from_numpy(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        return t(bad), t(crop)
