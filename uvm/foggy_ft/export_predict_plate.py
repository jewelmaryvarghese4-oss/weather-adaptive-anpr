"""
Read licence plates with LPRNet. Needs only: torch, pillow (and ultralytics + opencv if you give --detector).

  python predict_plate.py --input photo.jpg --detector detector/yolo_plate_detector_15k_best.pt     # whole photo: find plates, then read
  python predict_plate.py --input data/foggy_crops_test/crops --no_detect                           # a folder of plate crops
  python predict_plate.py --input photos_folder --detector detector/yolo_plate_detector_15k_best.pt --imgsz 1280 --csv out.csv

  python predict_plate.py --input data/foggy_crops_test/crops --no_detect --view select                # smoothing view + choose (see below)

--ckpt  weights/lprnet_avg_INSTALLED.pt (default) or any other file in weights/
--view  plain (default, unchanged behaviour) | smooth | select
          smooth : the crop is shrunk to 64x16 and enlarged to 128x32 (bicubic) before reading. This removes grain and helps on
                   blurry / noisy / foggy crops, but can hurt on sharp ones.
          select : read the plain crop AND the smoothed crop and keep the reading that looks like a plate (>= 8 characters, >= 4 digits,
                   >= 2 letters) with the higher confidence. No labels are used. Measured: +2 to +4 plates of 85 foggy crops for
                   readers that never saw those crops (see README).
"""
import argparse
import csv
import glob
import os
import sys

os.environ.setdefault("TWO_LINE_MODE", "fixed")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from lprnet_vocab import ctc_greedy_decode
from model_lprnet import LPRNet
from plate_format import ctc_format_constrained_decode
from two_line_plate import flatten_for_mode, normalize_plate_layout

INPUT_SIZE = (200, 24)           # (width, height) LPRNet is trained on
AMBIGUOUS_ASPECT = (1.9, 2.6)    # crops in this width/height range are read both as one line and as a flattened two-line plate


def looks_like_plate(text):
    return len(text) >= 8 and sum(c.isdigit() for c in text) >= 4 and sum(c.isalpha() for c in text) >= 2


@torch.no_grad()
def decode(net, img, dev):
    x = transforms.ToTensor()(img.resize(INPUT_SIZE)).unsqueeze(0).to(dev)
    lp = torch.log_softmax(net(x), dim=1).permute(2, 0, 1).cpu()
    return ctc_greedy_decode(lp)[0], ctc_format_constrained_decode(lp), lp[:, 0, :].max(dim=1).values.mean().item()


def read_plate(net, img, dev):
    """Returns (plain reading, reading corrected to the Indian plate format, confidence)."""
    img = img.convert("RGB")
    lo, hi = AMBIGUOUS_ASPECT
    cands = [img, flatten_for_mode(img)] if lo <= img.width / img.height < hi else [normalize_plate_layout(img)]
    best = None
    for c in cands:
        raw, fmt, conf = decode(net, c, dev)
        key = (looks_like_plate(fmt), conf)
        if best is None or key > best[0]:
            best = (key, raw, fmt, conf)
    return best[1], best[2], best[3]


def smooth_view(img):
    """Down-up resize (64x16 then 128x32, bicubic): a low-pass view of the crop."""
    return img.convert("RGB").resize((64, 16), Image.BICUBIC).resize((128, 32), Image.BICUBIC)


def read_plate_view(net, img, dev, view="plain"):
    """Like read_plate, with an optional smoothed view. Returns (plain reading, format-corrected reading, confidence, view used)."""
    if view == "plain":
        return read_plate(net, img, dev) + ("plain",)
    if view == "smooth":
        return read_plate(net, smooth_view(img), dev) + ("smooth",)
    a = read_plate(net, img, dev) + ("plain",)
    b = read_plate(net, smooth_view(img), dev) + ("smooth",)
    return max(a, b, key=lambda r: (looks_like_plate(r[1]), r[2]))


def load_net(ckpt, dev):
    net = LPRNet(num_classes=37).to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="an image, or a folder of images")
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "lprnet_avg_INSTALLED.pt"))
    ap.add_argument("--detector", default=None, help="YOLO plate detector weights; omit with --no_detect")
    ap.add_argument("--no_detect", action="store_true", help="the inputs are already plate crops")
    ap.add_argument("--imgsz", type=int, default=1280, help="detector input size (1280 finds more plates in large photos)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--view", choices=["plain", "smooth", "select"], default="plain", help="see the top of this file")
    a = ap.parse_args()
    if not a.no_detect and not a.detector:
        ap.error("give --detector, or --no_detect if the inputs are plate crops")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_net(a.ckpt, dev)
    det = None
    if not a.no_detect:
        from ultralytics import YOLO
        det = YOLO(a.detector)
    files = [a.input] if os.path.isfile(a.input) else sorted(f for f in glob.glob(os.path.join(a.input, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
    rows = []
    for f in files:
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")      # phone photos are often stored sideways
        crops = [(None, img)]
        if det is not None:
            r = det.predict(img, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            crops = [(b, img.crop(tuple(int(v) for v in b))) for b in r.boxes.xyxy.tolist()]
        for i, (box, c) in enumerate(crops):
            raw, fmt, conf, used = read_plate_view(net, c, dev, a.view)
            ok = looks_like_plate(fmt)
            if det is not None and not ok:
                continue                                                  # not a plate (sign, sticker, ...)
            rows.append((os.path.basename(f), i, "" if box is None else ",".join("%d" % v for v in box), fmt, raw, "%.3f" % conf, used))
            print("%-28s plate %d  %-12s (plain reading %s, confidence %.2f%s)" % (os.path.basename(f), i, fmt, raw, conf,
                                                                              ", view " + used if a.view == "select" else ""))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "plate", "box", "reading_format_corrected", "reading_plain", "confidence", "view_used"])
            w.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
