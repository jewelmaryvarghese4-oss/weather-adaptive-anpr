"""
evaluate_dehazing.py
---------------------
Runs EasyOCR over clear/, hazy/, dehazed_baseline/, and dehazed_v2/ folders
(matching filenames across all four), and reports average OCR confidence
per version - plus a separate breakdown on the glare-heavy subset.

Usage:
    python evaluate_dehazing.py --clear_dir ../datasets/dataset/clear --hazy_dir ../datasets/dataset/hazy \
        --baseline_dir ../dehazing_methods/dcp_dehazing/results/dataset_dehazed_baseline --v2_dir ../dehazing_methods/dcp_dehazing/results/dataset_dehazed_v2

Requires: pip install easyocr
"""

import argparse
import os
import re
import glob
import csv
import sys
import time

import numpy as np
import cv2

try:
    import easyocr
except ImportError:
    print("easyocr is not installed. Run: pip install easyocr")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dehazing_methods", "dcp_dehazing"))
from dehaze_v2 import build_glare_mask


def to_clear_filename(hazy_filename: str) -> str:
    """Maps 'crop_0000_v3.jpg' -> 'crop_0000.jpg' (strips _vN suffix)."""
    base, ext = os.path.splitext(hazy_filename)
    base = re.sub(r"_v\d+$", "", base)
    return base + ext


def list_images(folder: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(os.path.basename(p) for p in paths)


def get_ocr_confidence(reader, image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return None
    results = reader.readtext(img)
    if not results:
        return None
    confidences = [conf for (_bbox, _text, conf) in results]
    return float(np.mean(confidences))


def is_glare_heavy(image_path: str, v_thresh: int = 240, s_thresh: int = 30,
                    glare_fraction_thresh: float = 0.02) -> bool:
    img = cv2.imread(image_path)
    if img is None:
        return False
    mask = build_glare_mask(img, v_thresh, s_thresh)
    return (mask.sum() / mask.size) > glare_fraction_thresh


def evaluate_folder(reader, folder: str, filenames, name_transform=None):
    confidences = []
    skipped_no_text = 0
    missing = 0
    cache = {}

    for filename in filenames:
        lookup_name = name_transform(filename) if name_transform else filename
        path = os.path.join(folder, lookup_name)
        if not os.path.isfile(path):
            missing += 1
            continue

        if path in cache:
            conf = cache[path]
        else:
            conf = get_ocr_confidence(reader, path)
            cache[path] = conf

        if conf is None:
            skipped_no_text += 1
        else:
            confidences.append(conf)

    if missing:
        print(f"  [warning] {missing} file(s) referenced but missing from {folder}")

    avg_conf = float(np.mean(confidences)) if confidences else float("nan")
    return avg_conf, len(confidences), skipped_no_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear_dir", default="../datasets/dataset/clear")
    parser.add_argument("--hazy_dir", default="../datasets/dataset/hazy")
    parser.add_argument("--baseline_dir", default="../dehazing_methods/dcp_dehazing/results/dataset_dehazed_baseline")
    parser.add_argument("--v2_dir", default="../dehazing_methods/dcp_dehazing/results/dataset_dehazed_v2")
    parser.add_argument("--out_csv", default="ocr_evaluation_results.csv")
    parser.add_argument("--out_md", default="ocr_evaluation_results.md")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    print("Loading EasyOCR reader...")
    reader = easyocr.Reader(["en"], gpu=args.gpu)

    filenames = list_images(args.hazy_dir)
    if not filenames:
        print(f"No images found in {args.hazy_dir}")
        sys.exit(1)
    if args.limit:
        filenames = filenames[: args.limit]

    print(f"Evaluating {len(filenames)} images across 4 folders...\n")

    folders = {
        "Clear ground truth": args.clear_dir,
        "Hazy (untouched)": args.hazy_dir,
        "Dehazed (DCP baseline)": args.baseline_dir,
        "Dehazed (DCP + glare-aware, v2)": args.v2_dir,
    }

    results = {}
    for label, folder in folders.items():
        if not os.path.isdir(folder):
            print(f"[skipping] {label}: folder not found ({folder})")
            continue
        print(f"Scoring: {label} ({folder}) ...")
        transform = to_clear_filename if label == "Clear ground truth" else None
        start = time.perf_counter()
        avg_conf, num_scored, num_no_text = evaluate_folder(reader, folder, filenames, transform)
        elapsed = time.perf_counter() - start
        results[label] = {"avg_confidence": avg_conf, "num_scored": num_scored, "num_no_text": num_no_text}
        print(f"  avg confidence: {avg_conf:.4f}  (scored {num_scored}, no-text {num_no_text})  [{elapsed:.1f}s]\n")

    print("Identifying glare-heavy subset...")
    glare_filenames = [f for f in filenames if is_glare_heavy(os.path.join(args.hazy_dir, f))]
    print(f"Glare-heavy subset: {len(glare_filenames)} / {len(filenames)} images\n")

    glare_results = {}
    if glare_filenames:
        for label, folder in folders.items():
            if not os.path.isdir(folder):
                continue
            print(f"Scoring glare subset: {label} ...")
            transform = to_clear_filename if label == "Clear ground truth" else None
            avg_conf, num_scored, num_no_text = evaluate_folder(reader, folder, glare_filenames, transform)
            glare_results[label] = {"avg_confidence": avg_conf, "num_scored": num_scored, "num_no_text": num_no_text}
            print(f"  avg confidence: {avg_conf:.4f} (scored {num_scored})\n")
    else:
        print("No glare-heavy images found - skipping subset table.\n")

    with open(args.out_md, "w") as f:
        f.write("# OCR Confidence Evaluation Results\n\n")
        f.write(f"Full dataset: {len(filenames)} images\n\n")
        f.write("## Full dataset\n\n")
        f.write("| Version | Avg. confidence | Images scored | No-text images |\n")
        f.write("|---|---|---|---|\n")
        for label, r in results.items():
            f.write(f"| {label} | {r['avg_confidence']:.4f} | {r['num_scored']} | {r['num_no_text']} |\n")
        if glare_results:
            f.write(f"\n## Glare-heavy subset ({len(glare_filenames)} images)\n\n")
            f.write("| Version | Avg. confidence | Images scored | No-text images |\n")
            f.write("|---|---|---|---|\n")
            for label, r in glare_results.items():
                f.write(f"| {label} | {r['avg_confidence']:.4f} | {r['num_scored']} | {r['num_no_text']} |\n")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "version", "avg_confidence", "num_scored", "num_no_text"])
        for label, r in results.items():
            writer.writerow(["full", label, r["avg_confidence"], r["num_scored"], r["num_no_text"]])
        for label, r in glare_results.items():
            writer.writerow(["glare_heavy", label, r["avg_confidence"], r["num_scored"], r["num_no_text"]])

    print("=== SUMMARY ===")
    print(f"Full results written to {args.out_md} and {args.out_csv}")
    print("\nFull dataset:")
    for label, r in results.items():
        print(f"  {label:38s} {r['avg_confidence']:.4f}")
    if glare_results:
        print("\nGlare-heavy subset:")
        for label, r in glare_results.items():
            print(f"  {label:38s} {r['avg_confidence']:.4f}")


if __name__ == "__main__":
    main()