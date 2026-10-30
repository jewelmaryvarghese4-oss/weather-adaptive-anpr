"""
Real hazy photos with one labelled plate each, for the plate-aware dehazer loss. __getitem__ returns a window of the photo that contains the plate
(the photo is first scaled so the plate is about as wide as plates are in the 1536 px Ponmudi photos), the plate box inside the window, and the
plate text as class indices. Single-line plates only (a two-line plate needs a layout step that is not differentiable).
No horizontal flips (text), soft grain on half of the samples to resemble the phone photos.
"""
import csv
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lprnet_vocab import encode
from synth_rainfog import add_noise
from synth_ponmudi import ponmudi_degrade


class HazyPlateDataset(Dataset):
    def __init__(self, items_csv, photo_dir, split="train", win=256, samples=1500, width_range=(70, 160), augment=True, seed=0, synth_beta=None):
        self.items = []
        for r in csv.DictReader(open(items_csv, encoding="utf-8")):
            if r["split"] != split:
                continue
            x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            if (x2 - x1) / (y2 - y1) < 2.6:
                continue
            name = r["photo"] if "." in r["photo"] else r["photo"] + ".png"
            im = cv2.imread(os.path.join(photo_dir, name))
            self.items.append((im, (x1, y1, x2, y2), r["label"], r["photo"]))
        self.win, self.samples, self.wr, self.augment, self.seed, self.synth_beta = win, samples, width_range, augment, seed, synth_beta

    def __len__(self):
        return self.samples if self.augment else len(self.items)

    def __getitem__(self, idx):
        rng = random.Random() if self.augment else random.Random(self.seed * 1_000_003 + idx)
        im, (x1, y1, x2, y2), label, _ = self.items[rng.randrange(len(self.items)) if self.augment else idx]
        tw = rng.uniform(*self.wr) if self.augment else float(np.mean(self.wr))
        s = tw / (x2 - x1)
        img = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        x1, y1, x2, y2 = x1 * s, y1 * s, x2 * s, y2 * s
        W = self.win
        ph, pw = max(0, W - img.shape[0]), max(0, W - img.shape[1])
        if ph or pw:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        h, w = img.shape[:2]
        lo_x, hi_x = max(0, int(x2) + 6 - W), min(w - W, int(x1) - 6)
        lo_y, hi_y = max(0, int(y2) + 6 - W), min(h - W, int(y1) - 6)
        x0 = rng.randint(lo_x, hi_x) if hi_x >= lo_x else max(0, min(w - W, int((x1 + x2) / 2) - W // 2))
        y0 = rng.randint(lo_y, hi_y) if hi_y >= lo_y else max(0, min(h - W, int((y1 + y2) / 2) - W // 2))
        patch = img[y0:y0 + W, x0:x0 + W]
        if self.synth_beta is not None:                  # clear photo + the calibrated Ponmudi fog, rain and grain
            patch = ponmudi_degrade(patch, np.random.default_rng(rng.getrandbits(32)), beta=self.synth_beta, rain_prob=0.3)
        patch = patch.astype(np.float32) / 255.0
        if self.synth_beta is None and self.augment and rng.random() < 0.5:
            patch = add_noise(patch, np.random.default_rng(rng.getrandbits(32)))
        t = torch.from_numpy(cv2.cvtColor(np.clip(patch, 0, 1), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        box = torch.tensor([x1 - x0, y1 - y0, x2 - x0, y2 - y0], dtype=torch.float32)
        return t, box, encode(label), label


def plate_collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    boxes = torch.stack([b[1] for b in batch])
    targets = torch.tensor([i for b in batch for i in b[2]], dtype=torch.long)
    lengths = torch.tensor([len(b[2]) for b in batch], dtype=torch.long)
    return imgs, boxes, targets, lengths, [b[3] for b in batch]
