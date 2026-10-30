"""
How do the dehazers' outputs compare on the 50 test photos (1536 px)? Haze, contrast, saturation, brightness for the untouched photo and each
model's output, plus the share of photos whose brightness stays in the range of the CLEAR Ponmudi photos' look.   (run in WSL)
"""
import glob
import os

import cv2
import numpy as np

H = os.path.expanduser("~/dehaze-eval")


def metrics(bgr):
    f = bgr.astype(np.float32) / 255.0
    dark = float(cv2.erode(f.min(axis=2), np.ones((15, 15), np.uint8)).mean())
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return dark, float(g.std()), float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255.0), float(g.mean())


def avg(folder):
    out = []
    for f in sorted(glob.glob(H + "/data/in1536/*.png")):
        p = f if folder == "raw" else "%s/out/%s/%s" % (H, folder, os.path.basename(f))
        if not os.path.exists(p):
            return None
        im = cv2.imread(p)
        im = cv2.resize(im, None, fx=600 / im.shape[1], fy=600 / im.shape[1], interpolation=cv2.INTER_AREA)
        out.append(metrics(im))
    return np.mean(out, axis=0)


print("%-14s %8s %9s %11s %11s   (clear Ponmudi photos for reference: haze 0.18-0.29, contrast ~0.17, brightness 0.43-0.51)" % ("", "haze", "contrast", "saturation", "brightness"))
for name in ("raw", "uvm_old", "uvm_ft2", "uvm_ponmudi", "uvm_gf"):
    m = avg(name)
    if m is not None:
        print("%-14s %8.3f %9.3f %11.3f %11.3f" % (name, *m))
