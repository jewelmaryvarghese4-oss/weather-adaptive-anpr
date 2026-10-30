"""
Fog check on the whole foggy_photos folder: untouched vs the earlier fine-tuned UVM (foggy_photos_uvm_latest) vs the Ponmudi-trained UVM
(foggy_photos_uvm_ponmudi). Haze (dark channel, lower = clearer), contrast, saturation, brightness per photo; and three before/after sheets.
    python foggy_ft/ponmudi_folder_check.py
"""
import glob
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

D = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets")) + os.sep
OUT = D + "foggy_eval/ponmudi_folder_check/"
os.makedirs(OUT, exist_ok=True)


def small(path, w=700):
    im = cv2.imread(path)
    s = w / im.shape[1]
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def metrics(bgr):
    f = bgr.astype(np.float32) / 255.0
    dark = float(cv2.erode(f.min(axis=2), np.ones((15, 15), np.uint8)).mean())
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return dark, float(g.std()), float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255.0), float(g.mean())


rows = []
for f in sorted(glob.glob(D + "foggy_photos_upright/*.jpg")):
    stem = os.path.splitext(os.path.basename(f))[0]
    a = D + "foggy_photos_uvm_latest/%s_restored.png" % stem
    b = D + "foggy_photos_uvm_ponmudi/%s_restored.png" % stem
    if not (os.path.exists(a) and os.path.exists(b)):
        continue
    imgs = [small(f), small(a), small(b)]
    rows.append((stem, [metrics(i) for i in imgs], imgs))
M = np.array([[x for m in r[1] for x in m] for r in rows])          # 12 columns: raw(4) ft2(4) ponmudi(4)
names = ["haze (lower = clearer)", "contrast", "saturation", "brightness"]
print("photos checked: %d of 60" % len(rows))
print("%-24s %9s %12s %14s | clear Ponmudi photos" % ("average", "untouched", "UVM ft2", "UVM Ponmudi"))
ref = ["0.18-0.29", "~0.17", "0.23 (0.15-0.28)", "0.43-0.51"]
for i, n in enumerate(names):
    print("%-24s %9.3f %12.3f %14.3f | %s" % (n, M[:, i].mean(), M[:, 4 + i].mean(), M[:, 8 + i].mean(), ref[i]))
h_raw, h_ft2, h_pon = M[:, 0], M[:, 4], M[:, 8]
print("photos with LESS haze after the Ponmudi UVM than after the earlier UVM: %d of %d (mean gap %.3f)" % ((h_pon < h_ft2).sum(), len(rows), (h_ft2 - h_pon).mean()))
print("photos where haze fell versus untouched: ft2 %d of %d | Ponmudi %d of %d" % ((h_ft2 < h_raw).sum(), len(rows), (h_pon < h_raw).sum(), len(rows)))
print("fog remaining as a share of the untouched fog: ft2 %.0f%% | Ponmudi %.0f%%" % (100 * h_ft2.mean() / h_raw.mean(), 100 * h_pon.mean() / h_raw.mean()))
order = np.argsort(-h_raw)
for label, idx in (("densest fog (top third)", order[:20]), ("medium fog", order[20:40]), ("lightest fog", order[40:])):
    print("  %-26s untouched %.3f -> ft2 %.3f -> Ponmudi %.3f" % (label, h_raw[idx].mean(), h_ft2[idx].mean(), h_pon[idx].mean()))
with open(OUT + "per_photo.csv", "w") as fh:
    fh.write("photo,haze_untouched,haze_ft2,haze_ponmudi,brightness_untouched,brightness_ponmudi,saturation_ponmudi\n")
    for r, m in zip(rows, M):
        fh.write("%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n" % (r[0], m[0], m[4], m[8], m[3], m[11], m[10]))
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
        d.text((6, 3), "%s | untouched (haze %.2f) | UVM earlier (%.2f) | UVM Ponmudi (%.2f)" % (stem[9:], ms[0][0], ms[1][0], ms[2][0]), fill=(255, 255, 255), font=font)
        tiles.append(np.asarray(img))
    Image.fromarray(np.vstack(tiles)).save(OUT + "compare_%d.jpg" % (si + 1), quality=86)
print("saved sheets and per_photo.csv in", OUT)
