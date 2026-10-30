"""
Test-time averaging for the dehazer, on the 50 photos at 1536 px:  flip   = original + horizontal flip,  flip4 = + vertical flips,
ms = the photo at 1200 / 1536 / 2000 px (each result resized back to 1536 and averaged),  msflip = scales and flips together.
    python shoot_dehaze_tta.py --model uvm_gf --mode flip          -> ~/dehaze-eval/out/uvm_gf_flip/
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="uvm_gf")
ap.add_argument("--mode", required=True, choices=["flip", "flip4", "ms", "msflip"])
ap.add_argument("--scales", default="1200,1536,2000")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
H = os.path.expanduser("~/dehaze-eval")
CK = {"uvm_gf": "checkpoints_ponmudi_gf/uvmnet_combined_final.pt", "uvm_ponmudi": "checkpoints_ponmudi/uvmnet_combined_final.pt",
      "uvm_ft2": "checkpoints_rainfog2/uvmnet_combined_final.pt"}[a.model]
sys.path.insert(0, "/mnt/c/projects/jewwww/uvm")
import test_uvmnet as uvm
from model_uvmnet import UVMNet

dev = torch.device("cuda")
torch.set_grad_enabled(False)
net = UVMNet(n_channels=3, backend="cuda", uvmb_size=32).to(dev)
net.load_state_dict(torch.load("/mnt/c/projects/jewwww/uvm/" + CK, map_location=dev))
net.eval()
OUT = "%s/out/%s_%s" % (H, a.model, a.mode)
os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(H + "/data/in1536/*.png"))
if a.limit:
    files = files[:a.limit]
scales = [int(s) for s in a.scales.split(",")] if a.mode.startswith("ms") else [1536]
flips = {"flip": ["id", "h"], "flip4": ["id", "h", "v", "hv"], "ms": ["id"], "msflip": ["id", "h"]}[a.mode]


def flip(im, f):
    if f == "h":
        return im.transpose(Image.FLIP_LEFT_RIGHT)
    if f == "v":
        return im.transpose(Image.FLIP_TOP_BOTTOM)
    if f == "hv":
        return im.transpose(Image.ROTATE_180)
    return im


times = []
for i, f in enumerate(files):
    im = Image.open(f).convert("RGB")
    acc = np.zeros((im.height, im.width, 3), np.float64)
    n = 0
    t = time.time()
    for s in scales:
        base = im if s == 1536 else im.resize((round(im.width * s / 1536), round(im.height * s / 1536)), Image.LANCZOS)
        for fl in flips:
            out = flip(uvm.process_image(net, flip(base, fl), dev), fl)           # these flips are their own inverse
            if out.size != im.size:
                out = out.resize(im.size, Image.LANCZOS)
            acc += np.asarray(out, np.float64)
            n += 1
    Image.fromarray(np.clip(acc / n + 0.5, 0, 255).astype(np.uint8)).save("%s/%s" % (OUT, os.path.basename(f)))
    times.append(time.time() - t)
    if i % 10 == 0:
        print("%s %s %d/%d %.1fs (%d passes)" % (a.model, a.mode, i + 1, len(files), times[-1], n), flush=True)
print("DONE %s %s: mean %.1f s per photo" % (a.model, a.mode, np.mean(times)))
