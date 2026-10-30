"""
YOLOv11 plate detector + installed LPRNet on a folder of images (same functions and settings as exports/_stage/lprnet_export/predict_plate.py:
detector imgsz 1280, conf 0.25, format-corrected reading, 'select' view = plain or smoothed crop, whichever the reader prefers; detections that do not
read like a plate are dropped).  Writes, next to nothing else, into --out:
  annotated/<name>.png   the photo with the detected boxes and the plate text
  crops/<name>_<k>.png   every kept plate crop
  results.csv            file, plate, box, detector confidence, reading (format-corrected), plain reading, reader confidence, view used
  summary.txt            counts
    python real_scratch/run_anpr_folder.py --input /mnt/c/.../finalout/fogrealgopro --out /mnt/c/.../finalout/fogrealgopro_anpr     (WSL)
"""
import argparse
import csv
import glob
import os
import sys

EXPORT = "/mnt/c/projects/jewwww/exports/_stage/lprnet_export"
sys.path.insert(0, os.path.join(EXPORT, "src"))
sys.path.insert(0, EXPORT)
os.environ.setdefault("TWO_LINE_MODE", "fixed")

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

from predict_plate import load_net, looks_like_plate, read_plate_view

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default=os.path.join(EXPORT, "weights", "lprnet_avg_INSTALLED.pt"))
ap.add_argument("--detector", default=os.path.join(EXPORT, "detector", "yolo_plate_detector_15k_best.pt"))
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--view", default="select", choices=["plain", "smooth", "select"])
a = ap.parse_args()


def num_key(p):
    b = os.path.splitext(os.path.basename(p))[0]
    return (0, int(b)) if b.isdigit() else (1, b)


def main():
    from ultralytics import YOLO
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_net(a.ckpt, dev)
    det = YOLO(a.detector)
    os.makedirs(os.path.join(a.out, "annotated"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "crops"), exist_ok=True)
    files = sorted((f for f in glob.glob(os.path.join(a.input, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))), key=num_key)
    rows, n_det, n_drop, with_plate = [], 0, 0, 0
    for f in tqdm(files, desc="ANPR", unit="img", dynamic_ncols=True):
        name = os.path.splitext(os.path.basename(f))[0]
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        r = det.predict(img, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        boxes, confs = r.boxes.xyxy.tolist(), r.boxes.conf.tolist()
        canvas = img.copy()
        dr = ImageDraw.Draw(canvas)
        lw = max(3, img.width // 400)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(18, img.width // 45))
        kept = 0
        for i, (b, dc) in enumerate(zip(boxes, confs)):
            n_det += 1
            crop = img.crop(tuple(int(v) for v in b))
            raw, fmt, conf, used = read_plate_view(net, crop, dev, a.view)
            if not looks_like_plate(fmt):
                n_drop += 1
                continue
            kept += 1
            crop.save(os.path.join(a.out, "crops", "%s_%d.png" % (name, kept - 1)))
            rows.append([name + os.path.splitext(f)[1], kept - 1, ",".join("%d" % v for v in b), "%.3f" % dc, fmt, raw, "%.3f" % conf, used])
            dr.rectangle([b[0], b[1], b[2], b[3]], outline=(0, 255, 0), width=lw)
            label = "%s  (%.2f)" % (fmt, dc)
            tw = dr.textlength(label, font=font)
            ty = max(0, b[1] - font.size - 8)
            dr.rectangle([b[0], ty, b[0] + tw + 12, ty + font.size + 6], fill=(0, 0, 0))
            dr.text((b[0] + 6, ty + 2), label, fill=(0, 255, 0), font=font)
        with_plate += kept > 0
        canvas.save(os.path.join(a.out, "annotated", name + ".png"))
    with open(os.path.join(a.out, "results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "plate", "box_x1,y1,x2,y2", "detector_confidence", "reading_format_corrected", "reading_plain", "reader_confidence", "view_used"])
        w.writerows(rows)
    s = ("images: %d\nimages with at least one plate read: %d\nplates kept (read like a plate): %d\ndetections dropped (did not read like a plate): %d\n"
         "detector: %s (YOLO11, imgsz %d, conf %.2f)\nreader: %s (view: %s)\n") % (len(files), with_plate, len(rows), n_drop, os.path.basename(a.detector), a.imgsz, a.conf, os.path.basename(a.ckpt), a.view)
    open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8").write(s)
    print(s)
    print("ANPR_DONE")


if __name__ == "__main__":
    main()
