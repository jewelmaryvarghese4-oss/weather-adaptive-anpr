"""
Whole-photo pipeline scored with every way of reading: for each source (untouched photo / dehazed output, with or without the guided filter) and each
reader, exact plates out of the 85 verified crops with the plain crop, the smoothed view, and the plain-or-smoothed choice.
Needs ~/dehaze-eval/out/<source>/ and ~/dehaze-eval/data/items.json.   python shoot_score_select.py   (run in WSL)
"""
import json
import os
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

H = os.path.expanduser("~/dehaze-eval")
dev = torch.device("cuda")
torch.set_grad_enabled(False)
items = json.load(open(H + "/data/items.json"))
READERS = [("installed", "checkpoints_lprnet/lprnet_avg.pt"), ("weather seed 0", "checkpoints_compare_weather/fixed_s0/lprnet_avg.pt"),
           ("weather seed 1", "checkpoints_compare_weather/fixed_s1/lprnet_avg.pt"), ("weather seed 2", "checkpoints_compare_weather/fixed_s2/lprnet_avg.pt")]
SOURCES = ["raw", "uvm_ft2", "uvm_ft2+gf", "uvm_ponmudi", "uvm_ponmudi+gf"]


def box(a, r):
    return cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def gf(out, hazy, r=4, eps=1e-2):
    g = np.asarray(hazy.convert("L"), np.float32) / 255.0
    g = (g - g.min()) / max(g.max() - g.min(), 1e-3)
    o = np.asarray(out, np.float32) / 255.0
    res = []
    for c in range(3):
        p = o[..., c]
        mi, mp = box(g, r), box(p, r)
        a = (box(g * p, r) - mi * mp) / (box(g * g, r) - mi * mi + eps)
        res.append(box(a, r) * g + box(mp - a * mi, r))
    return Image.fromarray(np.clip(np.stack(res, -1) * 255, 0, 255).astype(np.uint8))


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


# crops per source (cut once)
crops = {}
for src in SOURCES:
    base, use_gf = (src[:-3], True) if src.endswith("+gf") else (src, False)
    photos = {}
    lst = []
    for it in items:
        p = it["photo"]
        if p not in photos:
            hazy = Image.open("%s/data/in1536/%s.png" % (H, p)).convert("RGB")
            im = hazy if base == "raw" else Image.open("%s/out/%s/%s.png" % (H, base, p)).convert("RGB")
            photos[p] = gf(im, hazy) if use_gf else im
        lst.append((it["label"], photos[p].crop(tuple(int(round(v)) for v in it["box"]))))
    crops[src] = lst
res = {}
for rname, ck in READERS:
    net = LPRNet(num_classes=37).to(dev)
    net.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
    net.eval()
    for src in SOURCES:
        a = b = c = 0
        for truth, im in crops[src]:
            ta, ca, va = read(net, im)
            tb, cb, vb = read(net, word(im))
            pick = max([(va, ca, ta), (vb, cb, tb)])[2]
            a += ta == truth
            b += tb == truth
            c += pick == truth
        res[(rname, src)] = (a, b, c)
print("exact plates of %d (plain / smoothed / chosen):" % len(items))
print("%-16s " % "reader" + " ".join("%-17s" % s for s in SOURCES))
for rname, _ in READERS:
    print("%-16s " % rname + " ".join("%-17s" % ("%d / %d / %d" % res[(rname, s)]) for s in SOURCES))
