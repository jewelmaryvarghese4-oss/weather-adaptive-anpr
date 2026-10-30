"""
Fog check on the whole foggy_photos folder with the GoPro-trained UVM (checkpoints_ponmudi_gopro), full size, guided filter r4 applied as in training:
untouched | filter-trained UVM (before the GoPro training) | GoPro-trained UVM. Haze (dark channel, lower = clearer), contrast, saturation, brightness per photo,
per_photo.csv, before/after sheets, and the final images saved in datasets/foggy_photos_uvm_gopro_gf/.
    python foggy_ft/gopro_folder_check.py        (WSL, from uvm/)
"""
import glob
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

UVM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, UVM)
from gf_head import guided_filter_head

D = os.path.join(os.path.dirname(UVM), "datasets") + os.sep
OUT = D + "foggy_eval/gopro_folder_check/"
FINAL = D + "foggy_photos_uvm_gopro_gf/"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FINAL, exist_ok=True)
torch.set_grad_enabled(False)


def tens(bgr):
    return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255.0


def gf(pred_bgr, hazy_bgr):
    o = guided_filter_head(tens(pred_bgr), tens(hazy_bgr), 4, 1e-2)[0].permute(1, 2, 0).clamp(0, 1).numpy()
    return cv2.cvtColor((o * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def small(img, w=700):
    s = w / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def metrics(bgr):
    f = bgr.astype(np.float32) / 255.0
    dark = float(cv2.erode(f.min(axis=2), np.ones((15, 15), np.uint8)).mean())
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return dark, float(g.std()), float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255.0), float(g.mean())


rows = []
for f in sorted(glob.glob(D + "foggy_photos_upright/*.jpg")):
    stem = os.path.splitext(os.path.basename(f))[0]
    a = D + "foggy_photos_uvm_gf/%s_restored.png" % stem
    b = D + "foggy_photos_uvm_gopro/%s_restored.png" % stem
    if not (os.path.exists(a) and os.path.exists(b)):
        continue
    raw, ra, rb = cv2.imread(f), cv2.imread(a), cv2.imread(b)
    if ra.shape != raw.shape or rb.shape != raw.shape:
        print("size mismatch, skipped:", stem, raw.shape, ra.shape, rb.shape)
        continue
    ga, gb = gf(ra, raw), gf(rb, raw)
    cv2.imwrite(FINAL + "%s_restored.png" % stem, gb)
    imgs = [small(raw), small(ga), small(gb)]
    rows.append((stem, [metrics(i) for i in imgs], imgs))
    print("done", stem, flush=True)
M = np.array([[x for m in r[1] for x in m] for r in rows])          # raw(4) before(4) gopro(4)
names = ["haze (lower = clearer)", "contrast", "saturation", "brightness"]
print("photos checked: %d of 60" % len(rows))
print("%-24s %9s %16s %14s | clear Ponmudi photos" % ("average", "untouched", "filter-trained", "GoPro-trained"))
ref = ["0.18-0.29", "~0.17", "0.23 (0.15-0.28)", "0.43-0.51"]
for i, n in enumerate(names):
    print("%-24s %9.3f %16.3f %14.3f | %s" % (n, M[:, i].mean(), M[:, 4 + i].mean(), M[:, 8 + i].mean(), ref[i]))
h_raw, h_b, h_g = M[:, 0], M[:, 4], M[:, 8]
print("photos with LESS haze after the GoPro-trained UVM than before it: %d of %d (mean gap %.3f)" % ((h_g < h_b).sum(), len(rows), (h_b - h_g).mean()))
print("photos where haze fell versus untouched: before %d of %d | GoPro %d of %d" % ((h_b < h_raw).sum(), len(rows), (h_g < h_raw).sum(), len(rows)))
print("fog remaining as a share of the untouched fog: before %.0f%% | GoPro %.0f%%" % (100 * h_b.mean() / h_raw.mean(), 100 * h_g.mean() / h_raw.mean()))
order = np.argsort(-h_raw)
n3 = len(order) // 3
for label, idx in (("densest fog (top third)", order[:n3]), ("medium fog", order[n3:2 * n3]), ("lightest fog", order[2 * n3:])):
    print("  %-26s untouched %.3f -> before %.3f -> GoPro %.3f" % (label, h_raw[idx].mean(), h_b[idx].mean(), h_g[idx].mean()))
with open(OUT + "per_photo.csv", "w") as fh:
    fh.write("photo,haze_untouched,haze_before,haze_gopro,brightness_untouched,brightness_gopro,saturation_gopro,contrast_gopro\n")
    for r, m in zip(rows, M):
        fh.write("%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n" % (r[0], m[0], m[4], m[8], m[3], m[11], m[10], m[9]))
font = None
for fp in ("C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
    try:
        font = ImageFont.truetype(fp, 24)
        break
    except OSError:
        pass
font = font or ImageFont.load_default()
picks = [order[1], order[len(order) // 2], order[-3], order[5]]
for si, grp in enumerate((picks[:2], picks[2:])):
    tiles = []
    for i in grp:
        stem, ms, imgs = rows[i]
        h = min(im.shape[0] for im in imgs)
        strip = np.hstack([cv2.resize(im, (700, h)) if im.shape[0] != h else im for im in imgs])
        img = Image.fromarray(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 2100, 34], fill=(0, 0, 0))
        d.text((6, 3), "%s | untouched (haze %.2f) | before GoPro (%.2f) | GoPro-trained (%.2f)" % (stem[9:], ms[0][0], ms[1][0], ms[2][0]), fill=(255, 255, 255), font=font)
        tiles.append(np.asarray(img))
    Image.fromarray(np.vstack(tiles)).save(OUT + "compare_%d.jpg" % (si + 1), quality=86)
print("saved sheets, per_photo.csv in", OUT, "and the final images in", FINAL)
