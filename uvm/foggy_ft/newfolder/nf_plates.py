"""
Plate reading on the new photos for every model output: YOLO finds the plates on each 1536 px photo; the same box (scaled) is cut from every model's output
and read by the installed reader and by the weather-trained reader (seed 1), plain and smoothed view, keeping the plausible/more confident one ("chosen").
Writes results/plates/{crops/*.png, plates.json}. There is no ground truth for these photos: the page shows what each model's crop reads as.
    HOME=/home/jewel/nfhome python foggy_ft/newfolder/nf_plates.py
"""
import glob
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
from ultralytics import YOLO

import read_plate as rp
from model_lprnet import LPRNet

R = "/mnt/c/projects/jewwww/datasets/newfoggyphotos/results"
OUT = R + "/plates"
os.makedirs(OUT + "/crops", exist_ok=True)
dev = torch.device("cuda")
torch.set_grad_enabled(False)
det = YOLO("/mnt/c/projects/jewwww/yolo_plate_detection/runs/detect/runs/detect/anpr_plate_model_15k/weights/best.pt")
READERS = [("installed", "checkpoints_lprnet/lprnet_avg.pt"), ("weather", "checkpoints_compare_weather/fixed_s1/lprnet_avg.pt")]
nets = []
for n, ck in READERS:
    net = LPRNet(num_classes=37).to(dev)
    net.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
    net.eval()
    nets.append((n, net))

# (id, label, kind, name, apply guided filter) -- same display rules as the gallery
SRC = [("orig", "Original photo", "in", "", False), ("clahe", "CLAHE", "s", "clahe", False), ("fogremoval", "FogRemoval", "s", "fogremoval", False),
       ("hazeflow", "HazeFlow (raw)", "s", "hazeflow", False), ("hazeflow_a", "HazeFlow (re-aligned, black edge)", "s", "hazeflow_a", False), ("hazeflow_b", "HazeFlow (re-aligned, edge filled)", "s", "hazeflow_b", False),
       ("snn6k", "DehazeSNN RESIDE-6k", "s", "snn_l-reside-6k", False), ("snnits", "DehazeSNN RESIDE-ITS", "s", "snn_l-reside-its", False), ("snnots", "DehazeSNN RESIDE-OTS", "s", "snn_l-reside-ots", False), ("snnrs", "DehazeSNN RS-Haze", "s", "snn_l-rs-haze", False),
       ("uvm_old", "UVM (installed model)", "s", "uvm_old", False), ("uvm_ft2", "UVM rain+fog fine-tune", "s", "uvm_ft2", False), ("uvm_pon", "UVM Ponmudi-trained", "s", "uvm_ponmudi", False),
       ("uvm_gf", "UVM filter-trained (+ filter)", "s", "uvm_gf", True), ("uvm_pa", "UVM plate-aware (+ filter)", "s", "uvm_pa", True), ("uvm_gp", "UVM GoPro-trained (+ filter)", "s", "uvm_gp", True), ("uvm_gpt", "UVM GoPro-trained + tint fix", "s", "uvm_gpt", False),
       ("tta_flip", "UVM filter-trained, flip average", "s", "uvm_gf_flip", True), ("tta_flip4", "UVM filter-trained, 4-way flip", "s", "uvm_gf_flip4", True), ("tta_ms", "UVM filter-trained, multi-scale", "s", "uvm_gf_ms", True), ("tta_msflip", "UVM filter-trained, multi-scale + flip", "s", "uvm_gf_msflip", True),
       ("f_ft2", "UVM rain+fog fine-tune (full size)", "f", "ft2", False), ("f_pon", "UVM Ponmudi-trained (full size)", "f", "pon", False), ("f_gf", "UVM filter-trained (full size, + filter)", "f", "gf", False),
       ("f_gp", "UVM GoPro-trained (full size, tiled)", "f", "gopro_gf", False), ("f_seam", "GoPro, seamless + tint fix", "f", "gopro_clean", False), ("f_seam0", "GoPro, seamless, no tint fix", "f", "gopro_clean_notint", False),
       ("f_hyb", "GoPro hybrid (no tint fix)", "f", "hybrid", False), ("f_hybt", "GoPro hybrid + tint fix", "f", "hybrid_tint", False)]


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


def chosen(net, im):
    a, b = read(net, im), read(net, word(im))
    return max([(a[2], a[1], a[0]), (b[2], b[1], b[0])])[2]


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    return inter / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter + 1e-9)


def load_variant(sid, kind, name, usegf, stem, hazy):
    if kind == "in":
        return hazy
    p = ("%s/s1536/%s/%s.png" % (R, name, stem)) if kind == "s" else ("%s/full/%s/%s_restored.png" % (R, name, stem))
    if not os.path.exists(p):
        return None
    im = Image.open(p).convert("RGB")
    return gf(im, hazy) if usegf else im


res = {"photos": []}
DET_CONF = 0.1
for f in sorted(glob.glob(R + "/in1536/*.png")):
    stem = os.path.splitext(os.path.basename(f))[0]
    hazy = Image.open(f).convert("RGB")
    ims = {}
    found = {}                                   # sid -> list of (box in 1536 coords, conf) the detector finds on THAT output
    for sid, label, kind, name, usegf in SRC:
        im = load_variant(sid, kind, name, usegf, stem, hazy)
        if im is None:
            continue
        ims[sid] = im
        sc = im.width / hazy.width
        r = det.predict(im, conf=DET_CONF, verbose=False)[0]
        found[sid] = [([v / sc for v in b], float(c)) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()) if (b[2] - b[0]) / sc >= 25]
    # plate groups: boxes from different outputs that overlap are the same plate; keep groups seen on >= 2 outputs or with confidence >= 0.4
    groups = []
    for sid, lst in found.items():
        for b, c in lst:
            for g in groups:
                if iou(g["box"], b) > 0.3:
                    g["members"].append((sid, b, c)); break
            else:
                groups.append({"box": b, "members": [(sid, b, c)]})
    keep = []
    for g in groups:
        bs = np.array([m[1] for m in g["members"]])
        g["box"] = [float(v) for v in np.median(bs, 0)]
        if len({m[0] for m in g["members"]}) >= 2 or max(m[2] for m in g["members"]) >= 0.4:
            keep.append(g)
    P = {"photo": stem, "plates": []}
    for pi, g in enumerate(keep):
        b = g["box"]
        entry = {"box": [round(v, 1) for v in b], "reads": {}, "found_on": sorted({m[0] for m in g["members"]})}
        for sid, im in ims.items():
            sc = im.width / hazy.width
            crop = im.crop(tuple(int(round(v * sc)) for v in b))
            crop.save("%s/crops/%s_%d_%s.png" % (OUT, stem, pi, sid))
            entry["reads"][sid] = {n: chosen(net, crop) for n, net in nets}
        P["plates"].append(entry)
    res["photos"].append(P)
    print(stem, "plates:", len(keep), "| detected per output (min/max):", min(len(v) for v in found.values()), max(len(v) for v in found.values()), flush=True)
res["sources"] = [{"id": s[0], "label": s[1]} for s in SRC]
json.dump(res, open(OUT + "/plates.json", "w"))
print("done")
