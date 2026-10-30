"""
Option 2, step 1 (separate from every earlier model): relative depth for every REAL clear image we can train with fog on.
Uses Depth Anything V2 Small (Hugging Face, depth-anything/Depth-Anything-V2-Small-hf). Writes
  datasets/depthfog/depth/<id>.npy      float16 disparity map in [0,1] (1 = nearest), long side 320 px
  datasets/depthfog/manifest.csv        id, image_path, group (gopro_train | gopro_val | ponmudi), depth_path
    python real_scratch/depthfog_cache.py            (run from uvm/, in WSL)
"""
import argparse
import csv
import hashlib
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "datasets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=os.path.join(ROOT, "real_scratch"))
    ap.add_argument("--out", default=os.path.join(ROOT, "depthfog"))
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--long_side", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "depth"), exist_ok=True)

    def rd(name):
        return list(csv.DictReader(open(os.path.join(a.real, name), encoding="utf-8")))

    items = []
    seen = set()
    for r in rd("gopro_train_pairs.csv"):
        if r["clear_path"] not in seen:
            seen.add(r["clear_path"]); items.append((r["clear_path"], "gopro_train"))
    for r in rd("gopro_val_pairs.csv"):
        if r["clear_path"] not in seen:
            seen.add(r["clear_path"]); items.append((r["clear_path"], "gopro_val"))
    for r in rd("ponmudi_clear.csv"):
        if r["path"] not in seen:
            seen.add(r["path"]); items.append((r["path"], "ponmudi"))
    if a.limit:
        items = items[:a.limit]
    print("images:", len(items), "| groups:", {g: sum(1 for _, x in items if x == g) for g in ("gopro_train", "gopro_val", "ponmudi")}, flush=True)

    dev = torch.device("cuda")
    proc = AutoImageProcessor.from_pretrained(a.model)
    net = AutoModelForDepthEstimation.from_pretrained(a.model).to(dev).half().eval()
    rows = []
    with torch.no_grad():
        for p, g in tqdm(items, desc="depth", unit="img", dynamic_ncols=True):
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            inp = proc(images=rgb, return_tensors="pt")["pixel_values"].to(dev).half()
            d = net(pixel_values=inp).predicted_depth[0].float().cpu().numpy()      # larger = nearer (relative disparity)
            s = a.long_side / max(bgr.shape[:2])
            d = cv2.resize(d, (round(bgr.shape[1] * s), round(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
            lo, hi = np.percentile(d, 1), np.percentile(d, 99)
            d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float16)
            i = hashlib.sha1(p.encode()).hexdigest()[:16]
            dp = os.path.join(a.out, "depth", i + ".npy")
            np.save(dp, d)
            rows.append([i, p, g, dp])
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "image_path", "group", "depth_path"])
        w.writerows(rows)
    print("DEPTH_DONE", len(rows), "->", a.out, flush=True)


if __name__ == "__main__":
    main()
