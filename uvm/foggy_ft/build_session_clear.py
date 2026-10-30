"""
Builds a training set of plate crops from the extra photos of the same fog session (datasets/gdrive_extra_photos).

    python foggy_ft/build_session_clear.py        (run in WSL from uvm/; needs the lprnet export folder for the reading tools)

Labels are NOT human-written: each plate is read by the installed reader in two views (plain crop and the 64x16->128x32 smoothed view). A crop is
accepted only if (a) the reading has a valid Indian plate format, (b) both views give the SAME string and (c) the reader is confident
(mean max log-prob >= --min_conf). Every plate whose text is within 2 edits of any plate in our verified foggy set (the test set) is excluded,
so the test plates never appear in training. At most --cap crops per plate identity are kept (largest crops first).

Writes datasets/session_clear/{crops/*.png, candidates.csv, accepted.csv, excluded_test_identities.csv, check_sheet_NN.png}
"""
import argparse
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UVM = os.path.dirname(HERE)
ROOT = os.path.dirname(UVM)
D = os.path.join(ROOT, "datasets")
EXPORT = os.path.join(ROOT, "exports", "_stage", "lprnet_export")
sys.path.insert(0, os.path.join(EXPORT, "src"))
sys.path.insert(0, EXPORT)
sys.path.insert(0, UVM)
os.environ.setdefault("TWO_LINE_MODE", "fixed")

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from plate_format import FORMATS_BY_LENGTH
from predict_plate import load_net, read_plate, smooth_view

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.join(D, "gdrive_extra_photos"))
ap.add_argument("--out", default=os.path.join(D, "session_clear"))
ap.add_argument("--min_conf", type=float, default=-0.15)
ap.add_argument("--cap", type=int, default=4)
ap.add_argument("--min_w", type=int, default=60, help="skip plate boxes narrower than this many pixels")
a = ap.parse_args()


def conforms(t):
    return any(len(f) == len(t) and all((c.isalpha() if k == "L" else c.isdigit()) for c, k in zip(t, f)) for f in FORMATS_BY_LENGTH.get(len(t), []))


def lev(x, y):
    p = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        q = [i]
        for j, cy in enumerate(y, 1):
            q.append(min(p[j] + 1, q[j - 1] + 1, p[j - 1] + (cx != cy)))
        p = q
    return p[-1]


os.makedirs(os.path.join(a.out, "crops"), exist_ok=True)
for f in glob.glob(os.path.join(a.out, "crops", "*")):
    os.remove(f)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = load_net(os.path.join(EXPORT, "weights", "lprnet_avg_INSTALLED.pt"), dev)
from ultralytics import YOLO
det = YOLO(os.path.join(EXPORT, "detector", "yolo_plate_detector_15k_best.pt"))

test_ids = {r["label"] for r in csv.DictReader(open(os.path.join(D, "foggy_ft", "foggy_labels_verified.csv"), encoding="utf-8"))
            if r["status"] in ("ok", "fix", "likely") and r["label"]}
print("test identities to keep out of training:", len(test_ids), flush=True)

cands = []
for f in sorted(glob.glob(os.path.join(a.src, "*.jpg"))):
    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    r = det.predict(img, conf=0.25, imgsz=1280, verbose=False)[0]
    for k, b in enumerate(r.boxes.xyxy.tolist()):
        x1, y1, x2, y2 = [int(v) for v in b]
        if x2 - x1 < a.min_w:
            continue
        crop = img.crop((x1, y1, x2, y2))
        r1 = read_plate(net, crop, dev)
        r2 = read_plate(net, smooth_view(crop), dev)
        cands.append(dict(photo=os.path.basename(f), k=k, box=(x1, y1, x2, y2), w=x2 - x1, h=y2 - y1, plain=r1[1], smooth=r2[1],
                          conf=min(r1[2], r2[2]), crop=crop))
print("detected plate boxes (>= %d px wide): %d" % (a.min_w, len(cands)), flush=True)

acc, rows = [], []
for i, c in enumerate(cands):
    why = ""
    if not conforms(c["plain"]):
        why = "not a valid plate format"
    elif c["plain"] != c["smooth"]:
        why = "plain and smoothed readings disagree"
    elif c["conf"] < a.min_conf:
        why = "reader not confident"
    elif min(lev(c["plain"], t) for t in test_ids) <= 2:
        why = "same plate as a test plate (kept out)"
    c["why"] = why
    rows.append((i, c["photo"], c["k"], c["w"], c["h"], c["plain"], c["smooth"], round(c["conf"], 3), why or "accepted"))
    if not why:
        acc.append(c)
with open(os.path.join(a.out, "candidates.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "photo", "box", "w", "h", "plain", "smooth", "conf", "status"])
    w.writerows(rows)
with open(os.path.join(a.out, "excluded_test_identities.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["photo", "reading", "closest_test_plate"])
    for c in cands:
        if c["why"].startswith("same plate"):
            w.writerow([c["photo"], c["plain"], min(test_ids, key=lambda t: lev(c["plain"], t))])

# crops of plates that ARE in our verified foggy set: the clear photo shows the same car, so the label is the VERIFIED text (not the reader's).
# Used only for folds in which that plate is not held out (see make_folds_session.py). Match = reading within 1 edit of exactly one verified plate.
ver = sorted(test_ids)
matched = []
for c in cands:
    if not c["why"].startswith("same plate"):
        continue
    d = sorted((lev(c["plain"], t), t) for t in ver)
    if d[0][0] <= 1 and d[1][0] > 1:
        c["identity"] = d[0][1]
        matched.append(c)
by_id = {}
for c in matched:
    by_id.setdefault(c["identity"], []).append(c)
kept_m = []
for ident, lst in by_id.items():
    kept_m += sorted(lst, key=lambda c: -c["w"])[:a.cap]
kept_m.sort(key=lambda c: (c["identity"], c["photo"]))
with open(os.path.join(a.out, "matched.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "text", "identity_verified", "photo", "reading_in_clear_photo", "conf"])
    for n, c in enumerate(kept_m):
        name = "sm%03d.png" % n
        c["crop"].save(os.path.join(a.out, "crops", name))
        c["mname"] = name
        w.writerow(["session_clear/crops/" + name, c["identity"], c["identity"], c["photo"], c["plain"], round(c["conf"], 3)])
print("matched to a verified foggy plate: %d boxes -> %d crops kept, %d plates" % (len(matched), len(kept_m), len(by_id)))

# keep up to --cap crops per plate text (largest first)
by = {}
for c in acc:
    by.setdefault(c["plain"], []).append(c)
kept = []
for text, lst in by.items():
    kept += sorted(lst, key=lambda c: -c["w"])[:a.cap]
kept.sort(key=lambda c: (c["plain"], c["photo"]))
with open(os.path.join(a.out, "accepted.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["path", "text", "verified", "group", "photo", "conf"])
    for n, c in enumerate(kept):
        name = "sc%03d.png" % n
        c["crop"].save(os.path.join(a.out, "crops", name))
        c["name"] = name
        w.writerow(["session_clear/crops/" + name, c["plain"], "0", "session_clear", c["photo"], round(c["conf"], 3)])
from collections import Counter
cnt = Counter(c["why"] or "accepted" for c in cands)
print("candidates by outcome:", dict(cnt))
print("kept for training: %d crops of %d plate texts (max %d per plate)" % (len(kept), len(by), a.cap))

# check sheets: crop + label, 30 per sheet
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 24)
sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
CW, CH, per = 480, 120, 30
for s in range(0, len(kept), per):
    chunk = kept[s:s + per]
    sheet = Image.new("RGB", (CW * 3, CH * 10), "white")
    d = ImageDraw.Draw(sheet)
    for j, c in enumerate(chunk):
        x, y = (j % 3) * CW, (j // 3) * CH
        im = c["crop"].copy()
        im.thumbnail((CW - 10, 70))
        sheet.paste(im, (x + 5, y + 4))
        d.text((x + 5, y + 76), c["plain"], fill=(0, 90, 0), font=font)
        d.text((x + 5, y + 102), "%s  #%s" % (c["photo"][9:15], c["name"][2:5]), fill=(120, 120, 120), font=sm)
        d.rectangle([x, y, x + CW - 1, y + CH - 1], outline=(210, 210, 210))
    sheet.save(os.path.join(a.out, "check_sheet_%02d.png" % (s // per + 1)))
print("check sheets:", (len(kept) + per - 1) // per)
sheet = Image.new("RGB", (CW * 3, CH * 10), "white")
d = ImageDraw.Draw(sheet)
for j, c in enumerate(kept_m[:30]):
    x, y = (j % 3) * CW, (j // 3) * CH
    im = c["crop"].copy()
    im.thumbnail((CW - 10, 70))
    sheet.paste(im, (x + 5, y + 4))
    d.text((x + 5, y + 76), c["identity"], fill=(0, 0, 160), font=font)
    d.text((x + 5, y + 102), "clear photo read: %s  %s" % (c["plain"], c["photo"][9:15]), fill=(120, 120, 120), font=sm)
    d.rectangle([x, y, x + CW - 1, y + CH - 1], outline=(210, 210, 210))
sheet.save(os.path.join(a.out, "check_sheet_matched.png"))
