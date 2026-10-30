"""
Turns the Zenodo "Number Plate Number Identification Dataset" (full vehicle
photos + an Excel sheet of true plate numbers, CC BY 4.0,
https://zenodo.org/records/13954136) into plate-crop/text pairs for LPRNet.

For each photo: run the plate detector, read every detection with the current
model, and keep the detection whose reading is CLOSEST to the sheet's label --
but only if it is within --max_dist edits. That does two jobs at once:
  * picks the right plate when a photo shows several vehicles, and
  * rejects wrong boxes / wrong labels (those end up far from the label).
Pairs the model already reads perfectly (dist 0) are easy; pairs at dist 1-4
are the clearly-same-plate cases the model gets partly wrong -- the useful ones.

Writes <out_dir>/crops/*.png and <out_dir>/manifest.csv (path,text,dist,conf,src),
which train_lprnet.py can mix in with --manifest.

Run from WSL2:
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python prepare_zenodo.py --root ../datasets/dataset10 --out_dir ../datasets/dataset10_crops
"""
import argparse
import collections
import csv
import os
import re
import xml.etree.ElementTree as ET
import zipfile

import torch
from PIL import Image
from ultralytics import YOLO

from model_lprnet import LPRNet
from lprnet_vocab import NUM_CLASSES
from read_plate import DEFAULT_YOLO_CKPT, read_plate
from train_lprnet import levenshtein

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_xlsx(path):
    """Minimal xlsx reader (no openpyxl needed): returns a list of {column: value}."""
    z = zipfile.ZipFile(path)
    strings = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si"):
            strings.append("".join(t.text or "" for t in si.iter(NS + "t")))
    sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]
    rows = []
    for row in ET.fromstring(z.read(sheet)).iter(NS + "row"):
        vals = {}
        for c in row.findall(NS + "c"):
            col = re.match(r"[A-Z]+", c.get("r")).group()
            v = c.find(NS + "v")
            if v is None:
                is_ = c.find(NS + "is")
                vals[col] = "".join(t.text or "" for t in is_.iter(NS + "t")) if is_ is not None else ""
            else:
                vals[col] = strings[int(v.text)] if c.get("t") == "s" else v.text
        rows.append(vals)
    return rows


def name_key(name):
    # sheet says 'Y39.png', disk has 'y (39).jpg' -> both become 'y39'
    return re.sub(r"[^a-z0-9]", "", os.path.splitext(name)[0].lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with images/ and number_plate.xlsx")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lprnet_ckpt", default="checkpoints_lprnet/lprnet_avg.pt")
    ap.add_argument("--yolo_ckpt", default=DEFAULT_YOLO_CKPT)
    ap.add_argument("--yolo_conf", type=float, default=0.3)
    ap.add_argument("--max_dist", type=int, default=4, help="max edits between reading and label to call it 'verified'")
    ap.add_argument("--trust_conf", type=float, default=0.5,
                    help="when unverified, keep the largest detection only if its confidence is at least this")
    ap.add_argument("--min_w", type=int, default=40)
    ap.add_argument("--min_h", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="only process N images spread over the set (for testing)")
    ap.add_argument("--exclude_gt", default="../evaluation/baseline_ocr/ground_truth.json",
                    help="drop plates whose text is already in this labeled set (some images here are the "
                         "SAME photos as the original dataset; training on them would leak validation scenes)")
    ap.add_argument("--exclude_test", default="../datasets/heldout_test/labels.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = read_xlsx(os.path.join(args.root, "number_plate.xlsx"))[1:]
    label = {name_key(r["A"]): re.sub(r"[^A-Z0-9]", "", (r.get("B") or "").upper())
             for r in rows if r.get("A")}
    files = {name_key(f): f for f in os.listdir(os.path.join(args.root, "images"))}
    keys = sorted(k for k in label if k in files and 5 <= len(label[k]) <= 12)
    if args.limit:  # evenly spread over the whole set, not just the first names
        keys = keys[:: max(1, len(keys) // args.limit)][: args.limit]
    print(f"{len(keys)} images with a usable label")

    import json
    seen_texts = set()
    for p in (args.exclude_gt, args.exclude_test):
        if p and os.path.exists(p):
            seen_texts |= set(json.load(open(p)).values())
    print(f"excluding plates whose text is already in {len(seen_texts)} known labels")

    def group(k):  # which sub-collection an image belongs to
        f = files[k].lower()
        if f.endswith(".jpg"):
            return "phone photo (y)"
        return "car photo (C)" if re.fullmatch(r"c\d+", k) else "state-named png"

    device = torch.device(args.device)
    lprnet = LPRNet(num_classes=NUM_CLASSES).to(device)
    lprnet.load_state_dict(torch.load(args.lprnet_ckpt, map_location=device))
    lprnet.eval()
    yolo = YOLO(args.yolo_ckpt)

    os.makedirs(os.path.join(args.out_dir, "crops"), exist_ok=True)
    stats = collections.Counter()
    dists = collections.Counter()
    by_group = collections.defaultdict(collections.Counter)
    out_rows = []
    for n, k in enumerate(keys, 1):
        img = Image.open(os.path.join(args.root, "images", files[k])).convert("RGB")
        res = yolo.predict(img, conf=args.yolo_conf, verbose=False)
        best = None      # detection whose reading is closest to the label
        largest = None   # biggest confident detection (fallback when the model can't confirm)
        for r in res:
            for xyxy, conf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                crop = img.crop(tuple(xyxy))
                if crop.width < args.min_w or crop.height < args.min_h:
                    continue
                raw, fmt = read_plate(lprnet, crop, device)
                d = min(levenshtein(raw, label[k]), levenshtein(fmt, label[k]))
                if best is None or (d, -conf) < (best[0], -best[1]):
                    best = (d, conf, crop)
                area = crop.width * crop.height
                if conf >= args.trust_conf and (largest is None or area > largest[0]):
                    largest = (area, conf, crop, d)
        g = group(k)
        if best is None:
            stats["no usable detection"] += 1
            by_group[g]["no detection"] += 1
            continue
        d, conf, crop = best
        verified = d <= args.max_dist and d <= len(label[k]) // 2
        if not verified:
            # The model can't confirm this label. In spot checks most of these were
            # the RIGHT plate that the model simply reads badly (tilted, narrow,
            # two-line) -- the most useful cases -- with ~1 in 16 a wrong box.
            if largest is None:
                stats["too far from label and no confident box"] += 1
                by_group[g]["rejected"] += 1
                continue
            _, conf, crop, d = largest
        if label[k] in seen_texts:
            stats["already in your labeled data (skipped to avoid leakage)"] += 1
            by_group[g]["duplicate of known"] += 1
            continue
        stats["accepted (verified)" if verified else "accepted (unverified: largest confident box)"] += 1
        by_group[g]["verified" if verified else "unverified"] += 1
        dists[d] += 1
        rel = f"crops/{k}.png"
        crop.save(os.path.join(args.out_dir, rel))
        out_rows.append((rel, label[k], int(verified), d, round(conf, 2), files[k], g))
        if n % 100 == 0:
            print(f"  {n}/{len(keys)} processed, {len(out_rows)} kept", flush=True)

    with open(os.path.join(args.out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "text", "verified", "dist", "conf", "src", "group"])
        w.writerows(out_rows)

    # Sample sheet of UNVERIFIED pairs so you can eyeball the label noise yourself.
    import random as _random
    from PIL import ImageDraw
    unver = [r for r in out_rows if r[2] == 0]
    _random.Random(0).shuffle(unver)
    pick = unver[:24]
    if pick:
        cw, ch, cols = 330, 110, 3
        sheet = Image.new("RGB", (cols * cw, ((len(pick) + cols - 1) // cols) * ch), (30, 30, 30))
        dr = ImageDraw.Draw(sheet)
        for i, r in enumerate(pick):
            im = Image.open(os.path.join(args.out_dir, r[0])).convert("RGB")
            im.thumbnail((cw - 8, ch - 24))
            x, y = (i % cols) * cw, (i // cols) * ch
            sheet.paste(im, (x + 4, y + 2))
            dr.text((x + 4, y + ch - 18), f"label = {r[1]}", fill=(255, 255, 0))
        sheet.save(os.path.join(args.out_dir, "unverified_sample.png"))

    print("\n=== result ===")
    for s, c in stats.items():
        print(f"  {s}: {c}")
    print("  kept pairs by edit distance between model reading and label:", dict(sorted(dists.items())))
    print("  by sub-collection:")
    for g, c in sorted(by_group.items()):
        print(f"    {g:<18} {dict(c)}")
    print(f"manifest: {os.path.join(args.out_dir, 'manifest.csv')}")


if __name__ == "__main__":
    main()
