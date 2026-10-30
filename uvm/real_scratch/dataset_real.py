"""
Real-only training samples (no synthetic fog anywhere).
  paired   : a GoProHazy hazy frame + its place-matched clear frame; same scale, same crop window, same flip in both
  unpaired : a real Ponmudi foggy photo / video frame (foggy), reference = a random real clear image (Ponmudi clear photo, or a GoPro clear frame)
Returns (hazy, reference, paired_flag), tensors in [0,1].
"""
import csv
import os
import random
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _rd(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def _imread(p):
    im = cv2.imread(p)
    if im is None:
        raise FileNotFoundError(p)
    return im


class RealDataset(Dataset):
    def __init__(self, root, crop=256, samples=8000, gopro_frac=0.6, seed=0, cache=48, photo_long=1600):
        self.crop, self.samples, self.gopro_frac, self.seed, self.photo_long = crop, samples, gopro_frac, seed, photo_long
        self.pairs = [(r["hazy_path"], r["clear_path"]) for r in _rd(os.path.join(root, "gopro_train_pairs.csv"))]
        self.hazy = [(r["path"], r["source"]) for r in _rd(os.path.join(root, "ponmudi_hazy.csv"))]
        self.pclear = [r["path"] for r in _rd(os.path.join(root, "ponmudi_clear.csv"))]
        self.gclear = sorted({c for _, c in self.pairs})
        assert self.pairs and self.hazy and self.pclear, "missing manifests: run extract_frames.py first"
        self._cache = OrderedDict()
        self._cache_n = cache

    def __len__(self):
        return self.samples

    def _load_small(self, p):
        """big photos are shrunk once to ~photo_long px and cached"""
        if p in self._cache:
            self._cache.move_to_end(p)
            return self._cache[p]
        im = _imread(p)
        s = self.photo_long / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        self._cache[p] = (im, min(s, 1.0))
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return self._cache[p]

    def _crop(self, im, rng, ys=None):
        S = self.crop
        h, w = im.shape[:2]
        if h < S or w < S:
            k = S / min(h, w) * 1.01
            im = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
            h, w = im.shape[:2]
        y0, x0 = rng.randint(0, h - S), rng.randint(0, w - S)
        return im[y0:y0 + S, x0:x0 + S], (y0, x0), im

    @staticmethod
    def _t(x):
        return torch.from_numpy(cv2.cvtColor(np.ascontiguousarray(x), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx):
        rng = random.Random(self.seed * 1_000_003 + idx * 7919 + random.getrandbits(20))
        S = self.crop
        if rng.random() < self.gopro_frac:                      # paired, GoPro 1080p
            hp, cp = rng.choice(self.pairs)
            h, c = _imread(hp), _imread(cp)
            s = rng.uniform(0.55, 1.0)
            h = cv2.resize(h, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            c = cv2.resize(c, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_AREA)
            hc, (y0, x0), h2 = self._crop(h, rng)
            if h2 is not h:                                      # (never for 1080p crops of <= 1080*0.55 px, kept for safety)
                c = cv2.resize(c, (h2.shape[1], h2.shape[0]))
            cc = c[y0:y0 + S, x0:x0 + S]
            if rng.random() < 0.5:
                hc, cc = hc[:, ::-1], cc[:, ::-1]
            return self._t(hc), self._t(cc), torch.tensor(True)
        # unpaired: real Ponmudi fog + a real clear reference
        p, src = rng.choice(self.hazy)
        im, sc0 = self._load_small(p)
        if src == "photo":                                       # photos are ~3000 px; the 1536 px working size is scale ~0.5 of them
            s = rng.uniform(0.6, 1.0) * 1.0                     # relative to the cached 1600 px copy (= 0.3 .. 0.53 of the original)
        else:
            s = rng.uniform(0.5, 1.0) * (1.0 if im.shape[0] < 1200 and im.shape[1] < 1200 else 1.0)
        imr = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im
        hc, _, _ = self._crop(imr, rng)
        if rng.random() < 0.5:
            hc = hc[:, ::-1]
        if rng.random() < 0.6:
            rp = rng.choice(self.pclear)
            rim, _ = self._load_small(rp)
            s2 = rng.uniform(0.6, 1.0)
        else:
            rim, s2 = _imread(rng.choice(self.gclear)), rng.uniform(0.55, 1.0)
        rimr = cv2.resize(rim, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA)
        rc, _, _ = self._crop(rimr, rng)
        if rng.random() < 0.5:
            rc = rc[:, ::-1]
        return self._t(hc), self._t(rc), torch.tensor(False)
