"""
Does the guided filter's radius matter with photo size? Full-size photos (3000-4000 px), outputs of the earlier UVM and of the Ponmudi UVM,
guided filter (guide = the untouched photo) with radius 4 / 8 / 16 / 32 px, then the 85 verified crops read with two readers.
    python gf_scale_test.py     (run in WSL)
"""
import csv
import os
import re
import sys

os.chdir("/mnt/c/projects/jewwww/uvm")
sys.path.insert(0, ".")
os.environ["TWO_LINE_MODE"] = "fixed"
import cv2
import numpy as np
import torch
from PIL import Image

import read_plate as rp
from model_lprnet import LPRNet

D = "/mnt/c/projects/jewwww/datasets/"
dev = torch.device("cuda")
torch.set_grad_enabled(False)
READERS = [("installed", "checkpoints_lprnet/lprnet_avg.pt"), ("weather seed 1", "checkpoints_compare_weather/fixed_s1/lprnet_avg.pt")]
nets = []
for n, ck in READERS:
    m = LPRNet(num_classes=37).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
    nets.append((n, m.eval()))
lab = [r for r in csv.DictReader(open(D + "foggy_ft/foggy_labels_verified.csv", encoding="utf-8")) if r["status"] in ("ok", "fix")]
boxes = {}
for blk in re.split(r"\n=== ", open(D + "foggy_eval/full_crop_pipeline/read_raw.txt", errors="ignore").read())[1:]:
    name = blk.split(" ===")[0]
    for m in re.finditer(r"plate (\d+)\s+box=\(([^)]*)\)", blk):
        boxes[(name, int(m.group(1)))] = [int(float(v)) for v in m.group(2).split(",")]
by_photo = {}
for r in lab:
    by_photo.setdefault(r["photo"], []).append((boxes[(r["photo"], int(r["plate"]))], r["label"]))


def bf(a, r):
    return cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def gfilter(out_rgb, guide, r, eps=1e-2):
    o = out_rgb.astype(np.float32) / 255.0
    res = []
    mi, vi = bf(guide, r), bf(guide * guide, r)
    for c in range(3):
        p = o[..., c]
        mp = bf(p, r)
        a = (bf(guide * p, r) - mi * mp) / (vi - mi * mi + eps)
        res.append(bf(a, r) * guide + bf(mp - a * mi, r))
    return np.clip(np.stack(res, -1) * 255, 0, 255).astype(np.uint8)


def word(im):
    return im.resize((64, 16), Image.BICUBIC).resize((128, 32), Image.BICUBIC)


def read(net, img):
    lo, hi = rp.AMBIGUOUS_ASPECT
    cands = [img, rp.flatten_for_mode(img)] if lo <= img.width / img.height < hi else [rp.normalize_plate_layout(img)]
    best = None
    for c in cands:
        _, fmt, conf = rp._decode(net, c, dev)
        key = (rp.looks_like_plate(fmt), conf)
        if best is None or key > best[0]:
            best = (key, fmt, conf)
    return best[1], best[2], best[0][0]


SRC = [("untouched", None), ("UVM earlier", "foggy_photos_uvm_latest"), ("UVM Ponmudi", "foggy_photos_uvm_ponmudi")]
RAD = [0, 4, 8, 16, 32]
res = {}
for photo, items in by_photo.items():
    stem = photo[:-4]
    hazy = cv2.cvtColor(cv2.imread(D + "foggy_photos_upright/" + photo), cv2.COLOR_BGR2RGB)
    g = cv2.cvtColor(hazy, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g = (g - g.min()) / max(g.max() - g.min(), 1e-3)
    for sname, folder in SRC:
        base = hazy if folder is None else cv2.cvtColor(cv2.imread(D + "%s/%s_restored.png" % (folder, stem)), cv2.COLOR_BGR2RGB)
        for r in ([0] if folder is None else RAD):
            im = base if r == 0 else gfilter(base, g, r)
            for (box, truth) in items:
                crop = Image.fromarray(im).crop(tuple(box))
                for rn, net in nets:
                    ta, ca, va = read(net, crop)
                    tb, cb, vb = read(net, word(crop))
                    pick = max([(va, ca, ta), (vb, cb, tb)])[2]
                    t = res.setdefault((sname, r, rn), [0, 0, 0])
                    t[0] += ta == truth
                    t[1] += tb == truth
                    t[2] += pick == truth
    print("done", photo, flush=True)
n = len(lab)
print("exact plates of %d at FULL size (plain / smoothed / chosen)" % n)
for rn, _ in nets:
    print("--- reader: " + rn)
    for sname, folder in SRC:
        for r in ([0] if folder is None else RAD):
            v = res[(sname, r, rn)]
            print("  %-12s guided filter radius %-8s %2d / %2d / %2d" % (sname, "none" if r == 0 else str(r) + " px", *v))
