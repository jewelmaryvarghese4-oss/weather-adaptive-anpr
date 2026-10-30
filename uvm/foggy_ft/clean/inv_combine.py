"""inverse combination: new model's full-resolution detail on the hybrid's low-frequency base  ->  new + (blur(hybrid) - blur(new))   (blur sigma 2.5% of the long side)
    python foggy_ft/clean/inv_combine.py --hybrid <folder of *_restored.png> --new <folder> --out <folder>
(same recipe as inv100 in combine_variants.py, with the folders given on the command line)"""
import argparse
import glob
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--hybrid", required=True)
ap.add_argument("--new", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def lowpass(img, sig_frac=0.025):
    h, w = img.shape[:2]
    k = 8
    sm = cv2.resize(img, (w // k, h // k), interpolation=cv2.INTER_AREA).astype(np.float32)
    sm = cv2.GaussianBlur(sm, (0, 0), sig_frac * max(h, w) / k)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_CUBIC)


os.makedirs(a.out, exist_ok=True)
n = 0
for f in sorted(glob.glob(os.path.join(a.hybrid, "*_restored.png"))):
    name = os.path.basename(f)
    h_ = cv2.imread(f).astype(np.float32)
    n_ = cv2.imread(os.path.join(a.new, name))
    if n_ is None or n_.shape != h_.shape:
        continue
    n_ = n_.astype(np.float32)
    out = n_ + (lowpass(h_) - lowpass(n_))
    cv2.imwrite(os.path.join(a.out, name), np.clip(out + 0.5, 0, 255).astype(np.uint8))
    n += 1
print("inverse combination:", n, "->", a.out)
