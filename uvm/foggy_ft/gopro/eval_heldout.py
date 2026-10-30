"""
Held-out GoPro frames (videos never trained on): hazy | filter-trained model before | GoPro-trained model | clear reference, plus no-reference numbers.
Needs the dehazed folders made by run_gopro_training.sh (test_uvmnet.py outputs *_restored.png). Guided filter r4 applied here, as in training.
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np
import torch

UVM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, UVM)
from gf_head import guided_filter_head

B = os.path.join(os.path.dirname(UVM), "datasets", "gopro_hazy")
OUT = os.path.join(B, "eval_heldout")
os.makedirs(OUT, exist_ok=True)
rows = [r for r in csv.DictReader(open(os.path.join(B, "pairs", "pairs.csv"), encoding="utf-8")) if r["split"] == "heldout"]


def load(p):
    return cv2.imread(p)


def gf(img_bgr, hazy_bgr):
    t = lambda x: torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255
    o = guided_filter_head(t(img_bgr), t(hazy_bgr), 4, 1e-2)[0].permute(1, 2, 0).clamp(0, 1).numpy()
    return cv2.cvtColor((o * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def stats(img):
    f = img.astype(np.float32) / 255
    dc = cv2.erode(f.min(2), np.ones((15, 15), np.uint8)).mean()
    y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255
    return dc, y.std(), hsv, y.mean()


names = ["hazy", "before (filter-trained)", "GoPro-trained", "clear reference"]
acc = {n: [] for n in names}
sheets = []
for r in rows:
    stem = r["hazy_file"].replace("_hazy.png", "")
    hz = load(os.path.join(B, "pairs", "heldout", r["hazy_file"]))
    cl = load(os.path.join(B, "pairs", "heldout", r["clear_file"]))
    b = load(os.path.join(B, "out_before", stem + "_hazy_restored.png"))
    a = load(os.path.join(B, "out_after", stem + "_hazy_restored.png"))
    if b is None or a is None:
        continue
    b, a = gf(b, hz), gf(a, hz)
    for n, im in zip(names, (hz, b, a, cl)):
        acc[n].append(stats(im))
    if len(sheets) < 6 and int(r["hazy_frame"]) % 2 == 0 or len(sheets) < 2:
        small = [cv2.resize(x, (480, 270), interpolation=cv2.INTER_AREA) for x in (hz, b, a, cl)]
        for im, n in zip(small, names):
            cv2.putText(im, n, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        sheets.append(np.hstack(small))
cv2.imwrite(os.path.join(OUT, "before_after_heldout.jpg"), np.vstack(sheets), [cv2.IMWRITE_JPEG_QUALITY, 92])
print("held-out GoPro frames:", len(acc["hazy"]), "(videos never trained on)")
print(f"{'':26s} {'haze':>6s} {'contrast':>9s} {'saturation':>11s} {'brightness':>11s}   (haze = dark channel; lower = clearer)")
for n in names:
    m = np.mean(acc[n], 0)
    print(f"{n:26s} {m[0]:6.3f} {m[1]:9.3f} {m[2]:11.3f} {m[3]:11.3f}")
