"""
Checks for the seamless + tint-fixed dehazing: seam score (largest brightness step across the image after dividing out the photo's own brightness
profile; tile seams show up as big steps), colour shift in dense fog (Lab b*, out minus in; warm = positive), haze/contrast, and a before/after sheet.
    python foggy_ft/clean/clean_check.py        (compares foggy_photos_uvm_gopro_gf [tiled] with foggy_photos_uvm_gopro_clean)
"""
import glob
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

D = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets")) + os.sep
OUT = D + "foggy_eval/clean_check/"
os.makedirs(OUT, exist_ok=True)


def load(p, w=1000):
    im = cv2.imread(p)
    if im is None:
        return None
    s = w / max(im.shape[:2])
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def seam(inp, out, k=4):
    yi = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY).astype(np.float32) + 8
    yo = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32) + 8
    best = 0.0
    for ax in (0, 1):
        r = yo.mean(ax) / yi.mean(ax)
        r = np.convolve(r, np.ones(5) / 5, mode="valid")
        best = max(best, float(np.abs(r[k:] - r[:-k]).max()) / float(np.median(r)))
    return best


def metrics(bgr):
    f = bgr.astype(np.float32) / 255
    dark = float(cv2.erode(f.min(2), np.ones((15, 15), np.uint8)).mean())
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    return dark, float(g.std()), float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255), float(g.mean())


def dense_b(inp, out):
    d = cv2.erode(inp.astype(np.float32).min(2) / 255, np.ones((9, 9), np.uint8))
    m = d > np.percentile(d, 70)
    bi = cv2.cvtColor(inp.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)[..., 2][m].mean()
    bo = cv2.cvtColor(out.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)[..., 2][m].mean()
    return bo - bi


rows = []
for f in sorted(glob.glob(D + "foggy_photos_upright/*.jpg")):
    st = os.path.splitext(os.path.basename(f))[0]
    raw, old, new = load(f), load(D + "foggy_photos_uvm_gopro_gf/%s_restored.png" % st), load(D + "foggy_photos_uvm_gopro_clean/%s_restored.png" % st)
    if old is None or new is None or old.shape != raw.shape or new.shape != raw.shape:
        continue
    rows.append((st, raw, old, new, seam(raw, old), seam(raw, new), dense_b(raw, old), dense_b(raw, new), metrics(old), metrics(new)))
S = np.array([[r[4], r[5], r[6], r[7]] + list(r[8]) + list(r[9]) for r in rows])
print("photos compared:", len(rows))
print("seam score (largest brightness step; tile seams make it big): tiled %.3f mean (max %.3f) -> seamless %.3f mean (max %.3f)" % (S[:, 0].mean(), S[:, 0].max(), S[:, 1].mean(), S[:, 1].max()))
print("photos where the step got smaller: %d of %d | photos with a clear seam before (score > 0.05): %d -> %d" % ((S[:, 1] < S[:, 0]).sum(), len(rows), (S[:, 0] > 0.05).sum(), (S[:, 1] > 0.05).sum()))
print("warm shift in dense fog (Lab b*, out - in): before %.1f -> after %.1f   (clear Ponmudi photos' whole-image b* is about 6)" % (S[:, 2].mean(), S[:, 3].mean()))
print("                      haze    contrast  saturation  brightness")
print("tiled (before)      %6.3f  %8.3f  %10.3f  %10.3f" % tuple(S[:, 4:8].mean(0)))
print("seamless + tint fix %6.3f  %8.3f  %10.3f  %10.3f" % tuple(S[:, 8:12].mean(0)))
with open(OUT + "per_photo.csv", "w") as fh:
    fh.write("photo,seam_before,seam_after,warm_before,warm_after\n")
    for r, s in zip(rows, S):
        fh.write("%s,%.3f,%.3f,%.1f,%.1f\n" % (r[0], s[0], s[1], s[2], s[3]))
font = None
for fp in ("C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
    try:
        font = ImageFont.truetype(fp, 22)
        break
    except OSError:
        pass
font = font or ImageFont.load_default()
order = np.argsort(-S[:, 0])               # worst seams first
picks = list(order[:3]) + [int(np.argmax(S[:, 2]))] + [len(rows) // 2]
tiles = []
for i in dict.fromkeys(picks):
    st, raw, old, new = rows[i][:4]
    strip = np.hstack([cv2.resize(x, (560, int(560 * x.shape[0] / x.shape[1]))) for x in (raw, old, new)])
    img = Image.fromarray(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1700, 30], fill=(0, 0, 0))
    d.text((6, 3), "%s | untouched | tiled (seam %.3f, warm %+.1f) | seamless + tint fix (seam %.3f, warm %+.1f)" % (st[9:], rows[i][4], rows[i][6], rows[i][5], rows[i][7]), fill=(255, 255, 255), font=font)
    tiles.append(np.asarray(img))
w = min(t.shape[1] for t in tiles)
Image.fromarray(np.vstack([t[:, :w] for t in tiles])).save(OUT + "before_after_clean.jpg", quality=88)
print("saved sheet + per_photo.csv in", OUT)
