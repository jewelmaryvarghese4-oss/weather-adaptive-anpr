"""
process_dataset.py
-------------------
Batch-runs dehaze_clahe.apply_clahe() over every image in a hazy folder.

CLAHE (Contrast Limited Adaptive Histogram Equalization) replaced the
earlier DCP-based pipeline (dehaze_baseline.py / dehaze_v2.py) here.
DCP models haze physically (atmospheric light + a transmission map) and
that model assumes the fog varies across the scene - it breaks down on
dense, roughly-uniform fog and was verified (on dataset2/New Hazy
dataset) to amplify whole frames to solid white instead of recovering
them. CLAHE has no physical haze model to violate: it just redistributes
local contrast per-tile, so it can't diverge that way, and it recovered
plate text DCP had destroyed. It's also ~13x faster (no atmospheric
light estimation or guided-filter refinement).

dehaze_baseline.py / dehaze_v2.py live in ../dcp_dehazing/ for reference
and are still used by evaluate_dehazing.py for comparison, but are no
longer part of this batch script.

Usage:
    python process_dataset.py --hazy_dir "../datasets/dataset2/New Hazy dataset" --out_clahe results/dataset2_dehazed_clahe
"""

import argparse
import os
import time
import glob
import sys

import cv2
import numpy as np

from dehaze_clahe import apply_clahe


def list_images(folder: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hazy_dir", default="hazy")
    parser.add_argument("--out_clahe", default="dehazed_clahe")
    parser.add_argument("--clahe_clip", type=float, default=3.0,
                         help="CLAHE clip limit (default: 3.0)")
    parser.add_argument("--clahe_grid", type=int, default=8,
                         help="CLAHE tile grid size, NxN (default: 8)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional: only process the first N images")
    args = parser.parse_args()

    if not os.path.isdir(args.hazy_dir):
        print(f"ERROR: hazy directory not found: {args.hazy_dir}")
        sys.exit(1)

    os.makedirs(args.out_clahe, exist_ok=True)

    image_paths = list_images(args.hazy_dir)
    if args.limit:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        print(f"No images found in {args.hazy_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images.")
    print(f"CLAHE params: clip_limit={args.clahe_clip}, tile_grid_size={args.clahe_grid}")

    clahe_times = []
    failed = []

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            failed.append(path)
            continue

        filename = os.path.basename(path)

        start = time.perf_counter()
        out_clahe = apply_clahe(img, clip_limit=args.clahe_clip, tile_grid_size=args.clahe_grid)
        clahe_times.append((time.perf_counter() - start) * 1000)

        cv2.imwrite(os.path.join(args.out_clahe, filename), out_clahe)

        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            print(f"  processed {i + 1}/{len(image_paths)}...")

    print("\n=== DONE ===")
    print(f"Images processed: {len(clahe_times)}")
    print(f"Failed to read:   {len(failed)}")
    if clahe_times:
        print(f"CLAHE latency: mean {np.mean(clahe_times):.2f} ms, "
              f"p95 {np.percentile(clahe_times, 95):.2f} ms")
    print(f"CLAHE outputs -> {args.out_clahe}/")
    if failed:
        print("\nFailed files:")
        for f in failed[:10]:
            print(f"  {f}")


if __name__ == "__main__":
    main()
