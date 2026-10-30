"""Shootout step 0 (CPU): upright photos resized to max side 1536, plus scaled plate boxes and verified labels."""
import csv
import json
import os
import re

from PIL import Image, ImageOps

D = "/mnt/c/projects/jewwww/datasets"
OUT = os.path.expanduser("~/dehaze-eval/data")
os.makedirs(OUT + "/in1536", exist_ok=True)
lab = {int(r["number"]): r for r in csv.DictReader(open(D + "/foggy_ft/foggy_labels_verified.csv", encoding="utf-8")) if r["status"] in ("ok", "fix")}
boxes = {}
for blk in re.split(r"\n=== ", open(D + "/foggy_eval/full_crop_pipeline/read_raw.txt", errors="ignore").read())[1:]:
    name = blk.split(" ===")[0]
    for m in re.finditer(r"plate (\d+)\s+box=\(([^)]*)\)", blk):
        boxes[(name, int(m.group(1)))] = [float(v) for v in m.group(2).split(",")]
photos = sorted({r["photo"] for r in lab.values()})
scale = {}
items = []
for p in photos:
    im = ImageOps.exif_transpose(Image.open("%s/foggy_photos/%s" % (D, p))).convert("RGB")
    s = min(1.0, 1536 / max(im.size))
    scale[p] = s
    im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS).save("%s/in1536/%s.png" % (OUT, os.path.splitext(p)[0]))
for n, r in lab.items():
    s = scale[r["photo"]]
    b = boxes[(r["photo"], int(r["plate"]))]
    items.append(dict(n=n, photo=os.path.splitext(r["photo"])[0], label=r["label"], box=[v * s for v in b]))
json.dump(items, open(OUT + "/items.json", "w"))
print("photos:", len(photos), "| verified plate crops:", len(items), "| scale example:", round(next(iter(scale.values())), 3))
