"""
postprocess_variants.py
------------------------
Cheap, no-retraining experiment: apply classical contrast/sharpness
post-processing on top of the current best Mamba checkpoint's output
(ckpt128_epoch50) and see whether it helps OCR legibility further.
Writes new folders that eval_metrics.py can score like any other config.
"""
import glob
import os

import cv2
import numpy as np

HERE = os.path.dirname(__file__)
SRC_DIR = os.path.join(HERE, "ckpt128_epoch50")


def clahe_variant(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def unsharp_variant(img_bgr, amount=1.0, sigma=1.0):
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    sharp = cv2.addWeighted(img_bgr, 1 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def clahe_then_unsharp(img_bgr):
    return unsharp_variant(clahe_variant(img_bgr), amount=0.6, sigma=1.0)


VARIANTS = {
    "ckpt128_epoch50_clahe": clahe_variant,
    "ckpt128_epoch50_unsharp": lambda im: unsharp_variant(im, amount=1.0, sigma=1.0),
    "ckpt128_epoch50_clahe_unsharp": clahe_then_unsharp,
}


def main():
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg")))
    print(f"{len(files)} source images from {SRC_DIR}")
    for name, fn in VARIANTS.items():
        out_dir = os.path.join(HERE, name)
        os.makedirs(out_dir, exist_ok=True)
        for fpath in files:
            img = cv2.imread(fpath)
            out = fn(img)
            cv2.imwrite(os.path.join(out_dir, os.path.basename(fpath)), out)
        print(f"  wrote {name} -> {out_dir}")


if __name__ == "__main__":
    main()
