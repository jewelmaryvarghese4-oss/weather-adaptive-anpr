"""All plate boxes (low threshold, so nothing is missed) on the clear Ponmudi photos -> datasets/ponmudi_clear/boxes.csv. Run in WSL from uvm/."""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
D = os.path.join(ROOT, "datasets")
EXPORT = os.path.join(ROOT, "exports", "_stage", "lprnet_export")
from PIL import Image, ImageOps
from ultralytics import YOLO

det = YOLO(os.path.join(EXPORT, "detector", "yolo_plate_detector_15k_best.pt"))
rows = []
for r in csv.DictReader(open(os.path.join(D, "ponmudi_clear", "list.csv"), encoding="utf-8")):
    img = ImageOps.exif_transpose(Image.open(os.path.join(D, "gdrive_extra_photos", r["file"]))).convert("RGB")
    res = det.predict(img, conf=0.05, imgsz=1280, verbose=False)[0]
    for b in res.boxes.xyxy.tolist():
        rows.append([r["file"]] + [round(v, 1) for v in b])
    print(r["file"], len(res.boxes), flush=True)
with open(os.path.join(D, "ponmudi_clear", "boxes.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "x1", "y1", "x2", "y2"])
    w.writerows(rows)
print("boxes:", len(rows))
