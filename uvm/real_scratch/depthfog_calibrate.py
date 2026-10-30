"""
Option 2, step 2: does the depth-aware fog look like the REAL Ponmudi fog? Compares dark-channel haze, contrast, saturation, brightness of
  real foggy photos (datasets/foggy_photos_upright, 60)  vs  depth-fog made on the clear Ponmudi photos with several thickness settings,
and writes a sheet (real fog on top, synthetic fog below) to datasets/depthfog/calibration_sheet.jpg.
    python real_scratch/depthfog_calibrate.py
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from depth_fog import depth_fog, load_depth

ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "datasets")
OUT = os.path.join(ROOT, "depthfog")


def stats(bgr):
    s = 768 / max(bgr.shape[:2])
    b = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
    f = b.astype(np.float32) / 255
    g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    return np.array([cv2.erode(f.min(2), np.ones((15, 15), np.uint8)).mean(), g.std(), cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255, g.mean()])


def line(name, m):
    m = np.array(m)
    print("%-34s haze %.3f (%.2f-%.2f)  contrast %.3f  sat %.3f  bright %.3f" % (name, m[:, 0].mean(), m[:, 0].min(), m[:, 0].max(), m[:, 1].mean(), m[:, 2].mean(), m[:, 3].mean()), flush=True)


def main():
    real = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(ROOT, "foggy_photos_upright", "*.jpg")))]
    line("REAL foggy Ponmudi photos (60)", [stats(r) for r in real])
    man = [r for r in csv.DictReader(open(os.path.join(OUT, "manifest.csv"), encoding="utf-8"))]
    pon = [r for r in man if r["group"] == "ponmudi"]
    gop = [r for r in man if r["group"] == "gopro_train"][::60]
    print("clear Ponmudi photos with depth:", len(pon), "| GoPro frames sampled:", len(gop), flush=True)
    settings = [("tau 0.8-4.5 near.15", (0.8, 4.5), (0.58, 0.90), 0.15), ("tau 1.2-5.5 near.15", (1.2, 5.5), (0.58, 0.90), 0.15), ("tau 0.8-4.5 near.25", (0.8, 4.5), (0.58, 0.90), 0.25), ("tau 1.2-5.5 near.25", (1.2, 5.5), (0.58, 0.90), 0.25)]
    sheet_rows = []
    for name, beta, arange, near in settings:
        for grp, items in (("Ponmudi", pon), ("GoPro", gop)):
            m, firsts = [], []
            for k, r in enumerate(items):
                im = cv2.imread(r["image_path"])
                s = 1024 / max(im.shape[:2])
                im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im
                d = load_depth(r["depth_path"], im.shape[:2])
                for j in range(3):
                    rng = np.random.default_rng(1000 * k + j)
                    f = depth_fog(im, d, rng, beta=beta, A_range=arange, near=near)
                    m.append(stats(f))
                    if k < 2 and j == 0:
                        firsts.append(f)
            line("synthetic %s on %s" % (name, grp), m)
            if grp == "Ponmudi" and name in ("tau 0.8-4.5 near.15", "tau 1.2-5.5 near.25"):
                sheet_rows.append(firsts)
    w = 360
    def tile(im):
        return cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])), interpolation=cv2.INTER_AREA)[:300]
    top = np.hstack([tile(real[i]) for i in (3, 20, 41, 55)])
    rows = [top]
    for fr in sheet_rows:
        rows.append(np.hstack([tile(x) for x in fr + fr[:2]][:4]))
    wmin = min(r.shape[1] for r in rows)
    cv2.imwrite(os.path.join(OUT, "calibration_sheet.jpg"), np.vstack([r[:, :wmin] for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("sheet ->", os.path.join(OUT, "calibration_sheet.jpg"))


if __name__ == "__main__":
    main()
