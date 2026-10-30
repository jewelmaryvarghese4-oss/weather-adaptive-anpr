"""
GoProHazy (DVD paper, CVPR 2024) real fog / clear road videos -> matched frame pairs, saved LOSSLESS (PNG, original 1920x1080, no resize).
The Drive's match files (datasets/gopro_hazy/match/N.txt) pair each hazy frame with the clear frame taken at the same place on the road; frame
numbers in them count every 3rd video frame (frame_k = video frame 3k, checked against the frame counts). The camera is NOT at exactly the same
position in the two passes, so the pairs are matched by place, not pixel-aligned: they are trained with a loss that tolerates that (train_uvm_gopro.py).
    python foggy_ft/gopro/extract_pairs.py [--per_video 30] [--hold 22 23]
Writes datasets/gopro_hazy/pairs/{train,heldout}/<n>_<k>_hazy.png + <n>_<k>_clear.png and pairs.csv.
"""
import argparse
import csv
import os
import re

import cv2
import numpy as np

B = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "datasets", "gopro_hazy")
ap = argparse.ArgumentParser()
ap.add_argument("--per_video", type=int, default=30)
ap.add_argument("--hold", type=int, nargs="*", default=[22, 23], help="videos kept out of training (for the before/after sheet and checks)")
ap.add_argument("--hold_per_video", type=int, default=12)
a = ap.parse_args()


def read_match(n):
    out = []
    for l in open(os.path.join(B, "match", f"{n}.txt"), encoding="utf-8"):
        h, c = l.strip().split("|")
        out.append((int(re.search(r"frame_(\d+)_hazy", h).group(1)), int(re.search(r"frame_(\d+)_clear", c).group(1))))
    return out


def grab(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    return f if ok else None


rows = []
videos = sorted(int(f.split("_")[0]) for f in os.listdir(os.path.join(B, "videos", "hazy")) if f.endswith("_hazy_video.mp4"))
for n in videos:
    hp, cp = (os.path.join(B, "videos", k, f"{n}_{k}_video.mp4") for k in ("hazy", "clear"))
    if not os.path.exists(os.path.join(B, "match", f"{n}.txt")) or not os.path.exists(cp):
        print("skip video", n, "(missing partner or match file)")
        continue
    split = "heldout" if n in a.hold else "train"
    pairs = read_match(n)
    # skip the first/last 5% (the car is starting / stopping), then take evenly spaced pairs; one pair per distinct clear frame
    lo, hi = int(len(pairs) * 0.05), int(len(pairs) * 0.95)
    pairs = pairs[lo:hi]
    k_want = a.hold_per_video if split == "heldout" else a.per_video
    pick = [pairs[int(i)] for i in np.linspace(0, len(pairs) - 1, k_want)]
    seen, hc, cc = set(), cv2.VideoCapture(hp), cv2.VideoCapture(cp)
    os.makedirs(os.path.join(B, "pairs", split), exist_ok=True)
    got = 0
    for k, m in pick:
        if m in seen:
            continue
        seen.add(m)
        h, c = grab(hc, 3 * k), grab(cc, 3 * m)
        if h is None or c is None:
            continue
        stem = os.path.join(B, "pairs", split, f"{n}_{k}")
        cv2.imwrite(stem + "_hazy.png", h)
        cv2.imwrite(stem + "_clear.png", c)
        rows.append([split, n, k, m, f"{n}_{k}_hazy.png", f"{n}_{k}_clear.png", h.shape[1], h.shape[0]])
        got += 1
    print(f"video {n:2d} ({split}): {got} pairs", flush=True)
with open(os.path.join(B, "pairs", "pairs.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["split", "video", "hazy_frame", "clear_frame", "hazy_file", "clear_file", "w", "h"])
    w.writerows(rows)
print("total pairs:", len(rows), "| train:", sum(r[0] == "train" for r in rows), "| heldout:", sum(r[0] == "heldout" for r in rows))
