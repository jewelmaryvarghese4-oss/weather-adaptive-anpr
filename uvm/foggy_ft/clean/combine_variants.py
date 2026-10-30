"""
Combinations of the GoPro hybrid and the new fresh model (epoch 22), written to separate folders (nothing existing is changed).
  blend70   : 0.7 * hybrid + 0.3 * new                       (pixel average)
  lf50      : hybrid + 0.5 * (blur(new) - blur(hybrid))      (hybrid's detail, half way to the new model's seamless low-frequency base)
  lf100     : hybrid + 1.0 * (blur(new) - blur(hybrid))      (hybrid's detail on the new model's low-frequency base)
  inv100    : new + 1.0 * (blur(hybrid) - blur(new))         (new model's detail on the hybrid's low-frequency base: the hybrid's fog removal, none of its sharp artifacts)
  inv70     : new + 0.7 * (blur(hybrid) - blur(new))
blur = Gaussian, sigma 2.5% of the long side (same as hybrid.py).
    python foggy_ft/clean/combine_variants.py
"""
import glob
import os
from multiprocessing import Pool

import cv2
import numpy as np

D = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets")) + os.sep
HYB, NEW = D + "foggy_photos_uvm_gopro_hybrid/", D + "foggy_photos_uvm_real/"
OUTS = {"blend70": D + "foggy_photos_uvm_hybrid_real_blend70/", "lf50": D + "foggy_photos_uvm_hybrid_real_lf50/", "lf100": D + "foggy_photos_uvm_hybrid_real_lf100/",
        "inv100": D + "foggy_photos_uvm_hybrid_real_inv100/", "inv70": D + "foggy_photos_uvm_hybrid_real_inv70/"}


def lowpass(img, sig_frac=0.025):
    h, w = img.shape[:2]
    k = 8
    sm = cv2.resize(img, (w // k, h // k), interpolation=cv2.INTER_AREA).astype(np.float32)
    sm = cv2.GaussianBlur(sm, (0, 0), sig_frac * max(h, w) / k)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_CUBIC)


def one(name):
    a = cv2.imread(HYB + name).astype(np.float32)
    b = cv2.imread(NEW + name).astype(np.float32)
    diff = lowpass(b) - lowpass(a)
    res = {"blend70": 0.7 * a + 0.3 * b, "lf50": a + 0.5 * diff, "lf100": a + 1.0 * diff, "inv100": b - 1.0 * diff, "inv70": b - 0.7 * diff}
    for k, v in res.items():
        cv2.imwrite(OUTS[k] + name, np.clip(v + 0.5, 0, 255).astype(np.uint8))
    return name


if __name__ == "__main__":
    for p in OUTS.values():
        os.makedirs(p, exist_ok=True)
    names = [os.path.basename(f) for f in sorted(glob.glob(HYB + "*_restored.png"))]
    with Pool(4) as pool:
        for i, _ in enumerate(pool.imap_unordered(one, names), 1):
            if i % 10 == 0:
                print(i, "/", len(names), flush=True)
    print("done", len(names))
