"""Shootout stage 2: read the verified plate crops from each model's outputs with the INSTALLED reader and count exact matches.
    python shoot_score.py name1 name2 ...      (names of folders in ~/dehaze-eval/out; 'raw' = the resized photo itself)
Optional suffix +gf = guided filter (guide = the hazy photo, r=4, eps=1e-2) applied to that model's output first."""
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
lpr = LPRNet(num_classes=37).to(dev)
lpr.load_state_dict(torch.load("checkpoints_lprnet/lprnet_avg.pt", map_location=dev))
lpr.eval()
torch.set_grad_enabled(False)
items = json.load(open(H + "/data/items.json"))


def read(img):
    img = img.convert("RGB")
    lo, hi = rp.AMBIGUOUS_ASPECT
    cands = [img, rp.flatten_for_mode(img)] if lo <= img.width / img.height < hi else [rp.normalize_plate_layout(img)]
    best = None
    for c in cands:
        _, fmt, conf = rp._decode(lpr, c, dev)
        key = (rp.looks_like_plate(fmt), conf)
        if best is None or key > best[0]:
            best = (key, fmt)
    return best[1]


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
        b = mp - a * mi
        res.append(box(a, r) * g + box(b, r))
    return Image.fromarray(np.clip(np.stack(res, -1) * 255, 0, 255).astype(np.uint8))


for name in sys.argv[1:]:
    base, use_gf = (name[:-3], True) if name.endswith("+gf") else (name, False)
    ok, per_photo = 0, {}
    for it in items:
        p = it["photo"]
        if p not in per_photo:
            hazy = Image.open("%s/data/in1536/%s.png" % (H, p)).convert("RGB")
            if base == "raw":
                im = hazy
            else:
                f = "%s/out/%s/%s.png" % (H, base, p)
                if not os.path.exists(f):
                    per_photo[p] = None
                    continue
                im = Image.open(f).convert("RGB")
                if use_gf:
                    im = gf(im, hazy)
            per_photo[p] = im
        im = per_photo[p]
        if im is None:
            continue
        b = [int(round(v)) for v in it["box"]]
        ok += read(im.crop(tuple(b))) == it["label"]
    n = sum(1 for it in items if per_photo.get(it["photo"]) is not None)
    print("%-26s exact %2d of %d crops (photos processed: %d)" % (name, ok, n, sum(v is not None for v in per_photo.values())), flush=True)
