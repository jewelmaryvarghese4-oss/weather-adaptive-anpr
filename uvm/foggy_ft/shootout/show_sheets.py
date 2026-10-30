import json
import os
import sys

os.chdir("/mnt/c/projects/jewwww/uvm")
sys.path.insert(0, ".")
os.environ["TWO_LINE_MODE"] = "fixed"
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import read_plate as rp
from model_lprnet import LPRNet

H = os.path.expanduser("~/dehaze-eval")
OUT = "/mnt/c/projects/jewwww/datasets/foggy_eval/dehazer_shootout/sheets"
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
lpr = LPRNet(num_classes=37).to(dev)
lpr.load_state_dict(torch.load("checkpoints_lprnet/lprnet_avg.pt", map_location=dev))
lpr.eval()
torch.set_grad_enabled(False)
items = json.load(open(H + "/data/items.json"))
try:
    F1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    F2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 30)
except Exception:
    F1 = F2 = ImageFont.load_default()


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


MODELS = [("Original", "raw"), ("CLAHE", "clahe"), ("UVM old", "uvm_old"), ("UVM fine-tuned", "uvm_ft2"), ("FogRemoval", "fogremoval"),
          ("DehazeSNN (ITS)", "snn_l-reside-its"), ("FogRemoval + guided filter", "fogremoval+gf"), ("UVM fine-tuned + guided filter", "uvm_ft2+gf")]
extra = sys.argv[1:]                                   # e.g. hazeflow
for e in extra:
    MODELS.append((e, e))


def get(photo, key):
    hazy = Image.open("%s/data/in1536/%s.png" % (H, photo)).convert("RGB")
    if key == "raw":
        return hazy
    base, usegf = (key[:-3], True) if key.endswith("+gf") else (key, False)
    f = "%s/out/%s/%s.png" % (H, base, photo)
    if not os.path.exists(f):
        return None
    im = Image.open(f).convert("RGB")
    return gf(im, hazy) if usegf else im


def sheet(photo, region, labels, name, cols=4, tw=520):
    tiles = []
    for title, key in MODELS:
        im = get(photo, key)
        if im is None:
            continue
        txt = ""
        if labels:
            ok = [read(im.crop(tuple(int(v) for v in b))) for b, _ in labels]
            txt = "  ".join("%s %s" % (r, "OK" if r == t else "x") for r, (b, t) in zip(ok, labels))
        c = im.crop(region)
        s = tw / c.width
        c = c.resize((tw, int(c.height * s)), Image.LANCZOS)
        tiles.append((title, c, txt))
    th = tiles[0][1].height
    rows = (len(tiles) + cols - 1) // cols
    cell_h = th + 44 + (40 if labels else 0)
    sh = Image.new("RGB", (cols * tw, rows * cell_h), "white")
    d = ImageDraw.Draw(sh)
    for i, (title, c, txt) in enumerate(tiles):
        x, y = (i % cols) * tw, (i // cols) * cell_h
        d.text((x + 8, y + 6), title, fill=(0, 0, 0), font=F1)
        sh.paste(c, (x, y + 44))
        if txt:
            d.text((x + 8, y + 44 + th + 4), txt[:36], fill=(0, 100, 0), font=F2)
    sh.save("%s/%s.jpg" % (OUT, name), quality=88)
    print("wrote", name, len(tiles), "tiles")


# scene A: rainy foggy photo with the Kia KL35N8421 and others; scene B: photo with three plates, clearer
for tag, photo in (("A", "20260926_111507"), ("B", "20260926_112417")):
    its = [it for it in items if it["photo"] == photo]
    labels = [(it["box"], it["label"]) for it in its]
    im0 = Image.open("%s/data/in1536/%s.png" % (H, photo))
    sheet(photo, (0, 0, im0.width, im0.height), None, "scene%s_full_view" % tag, cols=4, tw=480)
    # zoom: region around the first labelled plates
    b = its[0]["box"]
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    x0, y0 = int(max(0, cx - 200)), int(max(0, cy - 130))
    region = (x0, y0, min(im0.width, x0 + 400), min(im0.height, y0 + 260))
    lab1 = [(its[0]["box"], its[0]["label"])]
    sheet(photo, region, lab1, "scene%s_plate_zoom_%s" % (tag, its[0]["label"]), cols=4, tw=520)
