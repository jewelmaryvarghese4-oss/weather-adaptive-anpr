"""
Hybrid: the DETAIL of the old tiled result (full resolution, sharp) + the seam-free LOW-FREQUENCY base of the single-pass result.
    hybrid = tiled + ( blur(single_pass) - blur(tiled) )        blur = Gaussian, sigma ~ 2.5% of the long side (larger than the 128 px tile overlap)
The per-tile brightness/haze steps are low-frequency, so they are replaced by the single pass's consistent ones; edges, texture and local contrast stay the tiled ones.
Optional tint fix (tint_fix.py) is applied on top.   python foggy_ft/clean/hybrid.py --tint 0.0 --output ../datasets/foggy_photos_uvm_gopro_hybrid
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tint_fix import fix_tint

ap = argparse.ArgumentParser()
ap.add_argument("--tint", type=float, default=0.0)
ap.add_argument("--output", required=True)
ap.add_argument("--sigma_frac", type=float, default=0.025)
ap.add_argument("--hazy_dir", default=None)
ap.add_argument("--tiled_dir", default=None)
ap.add_argument("--seam_dir", default=None)
a = ap.parse_args()
D = os.path.abspath(os.path.join(HERE, "..", "..", "..", "datasets")) + os.sep
os.makedirs(a.output, exist_ok=True)
HAZY = a.hazy_dir or D + "foggy_photos_upright"
TILED = a.tiled_dir or D + "foggy_photos_uvm_gopro_gf"
SEAM = a.seam_dir or D + "foggy_photos_uvm_gopro_clean_notint"


def lowpass(img, sig_frac):
    h, w = img.shape[:2]
    k = 8
    sm = cv2.resize(img, (w // k, h // k), interpolation=cv2.INTER_AREA).astype(np.float32)
    sm = cv2.GaussianBlur(sm, (0, 0), sig_frac * max(h, w) / k)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_CUBIC)


n = 0
for f in sorted(glob.glob(HAZY + "/*.jpg") + glob.glob(HAZY + "/*.png")):
    st = os.path.splitext(os.path.basename(f))[0]
    hazy = cv2.imread(f)
    old = cv2.imread(TILED + "/%s_restored.png" % st)
    new = cv2.imread(SEAM + "/%s_restored.png" % st)
    if old is None or new is None or old.shape != new.shape or old.shape != hazy.shape:
        continue
    out = old.astype(np.float32) + (lowpass(new, a.sigma_frac) - lowpass(old, a.sigma_frac))
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    if a.tint > 0:
        out = fix_tint(hazy, out, a.tint)
    cv2.imwrite(os.path.join(a.output, st + "_restored.png"), out)
    n += 1
print("hybrid images:", n)
