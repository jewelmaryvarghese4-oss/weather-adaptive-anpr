"""
Items for the plate-aware dehazer loss: real hazy photos (Foggy-Hazy set) + the box of one labelled plate + its verified text.
    python foggy_ft/build_hazy_plate_items.py     (run in WSL from uvm/)
train items : photos whose plate crop is in the training manifest (hazy250_crops, verified text); the photos of the held-out plates are removed
test items  : the held-out plates of heldout_rj250 (never trained on), located in the hazy photo
Boxes are found with the same detector and settings that cut the original crops; a detection is accepted only if its size matches the saved crop.
Writes datasets/plate_aware/items.csv
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UVM = os.path.dirname(HERE)
ROOT = os.path.dirname(UVM)
D = os.path.join(ROOT, "datasets")
sys.path.insert(0, UVM)
from PIL import Image
from ultralytics import YOLO

det = YOLO(os.path.join(ROOT, "yolo_plate_detection", "runs", "detect", "runs", "detect", "anpr_plate_model_15k", "weights", "best.pt"))
PH = os.path.join(D, "uvmresults", "hazy_subset_250")
manifest = [r for r in csv.DictReader(open(os.path.join(D, "final_manifest_rj250_renamed.csv"), encoding="utf-8")) if r["path"].startswith("hazy250_crops/hazy/")]
held = json.load(open(os.path.join(D, "heldout_rj250", "labels.json")))
held_ids = {k.replace("_uvm", "") for k in held}
print("training hazy crops in the manifest:", len(manifest), "| held-out photos:", len(held_ids))


def detections(photo_id):
    p = os.path.join(PH, photo_id + ".png")
    if not os.path.exists(p):
        return None, []
    im = Image.open(p).convert("RGB")
    r = det.predict(im, conf=0.25, verbose=False)[0]
    return im, [tuple(b) for b in r.boxes.xyxy.tolist()]


rows, skipped = [], {"no photo": 0, "no size match": 0, "held-out photo": 0}
for r in manifest:
    pid = os.path.splitext(os.path.basename(r["path"]))[0]
    if pid in held_ids:
        skipped["held-out photo"] += 1
        continue
    cw, ch = Image.open(os.path.join(D, r["path"])).size
    im, bs = detections(pid)
    if im is None:
        skipped["no photo"] += 1
        continue
    best = min(bs, key=lambda b: abs((b[2] - b[0]) - cw) + abs((b[3] - b[1]) - ch), default=None)
    if best is None or abs((best[2] - best[0]) - cw) > 2 or abs((best[3] - best[1]) - ch) > 2:
        skipped["no size match"] += 1
        continue
    row = ["train", pid, *[round(v, 1) for v in best], r["text"], im.width, im.height]
    if row not in rows:                      # the manifest repeats each crop 4 times
        rows.append(row)
print("train items:", len(rows), "| skipped:", skipped)
nt = 0
for k, text in held.items():
    pid = k.replace("_uvm", "")
    crop = os.path.join(D, "heldout_rj250", "crops", k + ".png")
    cw, ch = Image.open(crop).size
    im, bs = detections(pid)
    if im is None or not bs:
        continue
    best = min(bs, key=lambda b: abs((b[2] - b[0]) - cw) + abs((b[3] - b[1]) - ch))
    if abs((best[2] - best[0]) - cw) > 6 or abs((best[3] - best[1]) - ch) > 4:
        continue
    rows.append(["test", pid, *[round(v, 1) for v in best], text, im.width, im.height])
    nt += 1
print("test items:", nt, "of", len(held))
os.makedirs(os.path.join(D, "plate_aware"), exist_ok=True)
with open(os.path.join(D, "plate_aware", "items.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["split", "photo", "x1", "y1", "x2", "y2", "label", "photo_w", "photo_h"])
    w.writerows(rows)
tr = [r for r in rows if r[0] == "train"]
asp = [(r[4] - r[2]) / (r[5] - r[3]) for r in tr]
print("train plates: %d distinct texts | single-line (aspect >= 2.6): %d | plate width px median %d" % (
    len({r[6] for r in tr}), sum(a >= 2.6 for a in asp), sorted(r[4] - r[2] for r in tr)[len(tr) // 2]))


# ---- the CLEAR photos of the same plates (separate shots): their boxes and verified texts, for training with synthetic fog on top
import glob
CL = os.path.join(D, "1001hazeimages", "FoggyPlates", "clear")
cm = [r for r in csv.DictReader(open(os.path.join(D, "final_manifest_rj250_renamed.csv"), encoding="utf-8")) if r["path"].startswith("hazy250_crops/clear/")]
crow = []
for r in cm:
    pid = os.path.splitext(os.path.basename(r["path"]))[0]
    if pid in held_ids:
        continue
    cands = glob.glob(os.path.join(CL, pid + ".*"))
    if not cands:
        continue
    im = Image.open(cands[0]).convert("RGB")
    cw, ch = Image.open(os.path.join(D, r["path"])).size
    bs = [tuple(b) for b in det.predict(im, conf=0.25, verbose=False)[0].boxes.xyxy.tolist()]
    best = min(bs, key=lambda b: abs((b[2] - b[0]) - cw) + abs((b[3] - b[1]) - ch), default=None)
    if best is None or abs((best[2] - best[0]) - cw) > 2 or abs((best[3] - best[1]) - ch) > 2:
        continue
    row = ["train", os.path.basename(cands[0]), *[round(v, 1) for v in best], r["text"], im.width, im.height]
    if row not in crow:
        crow.append(row)
with open(os.path.join(D, "plate_aware", "items_clear.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["split", "photo", "x1", "y1", "x2", "y2", "label", "photo_w", "photo_h"])
    w.writerows(crow)
print("clear-photo items for synthetic fog:", len(crow), "| single-line:", sum((r[4] - r[2]) / (r[5] - r[3]) >= 2.6 for r in crow))
