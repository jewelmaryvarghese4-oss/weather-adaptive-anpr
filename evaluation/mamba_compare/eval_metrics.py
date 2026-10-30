"""
eval_metrics.py
----------------
Compares dehazed-output folders (produced by generate_outputs.py in the
WSL mamba-project) against the untouched hazy and clear reference crops:

  - PSNR / SSIM vs ground-truth clear crop (image-quality)
  - Laplacian variance, both at native resolution and normalized to a
    common 128x128 scale (raw Laplacian variance isn't comparable between
    a 64x64 and a 128x128 image -- the discrete operator's response
    depends on pixel density, not just true edge sharpness -- so the
    normalized column is what should be used to compare across the
    64px vs 128px checkpoint families)
  - EasyOCR: average confidence, exact-match rate, and character-level
    accuracy (1 - normalized edit distance) against the real plate-text
    labels in ../baseline_ocr/ground_truth.json

Usage:
    python eval_metrics.py
"""
import glob
import json
import os
import re

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import easyocr

HERE = os.path.dirname(__file__)
GT_PATH = os.path.join(HERE, "..", "baseline_ocr", "ground_truth.json")
NORM_SIZE = 128


def normalize_text(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def edit_distance(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def char_accuracy(pred, gt):
    pred_n, gt_n = normalize_text(pred), normalize_text(gt)
    if not gt_n:
        return None
    dist = edit_distance(pred_n, gt_n)
    return max(0.0, 1.0 - dist / max(len(gt_n), 1))


def laplacian_var(img_bgr, resize_to=None):
    if resize_to is not None:
        img_bgr = cv2.resize(img_bgr, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def score_folder(files, clear_dir, gt, reader, allowlist):
    def ocr_top(path):
        out = reader.readtext(path, allowlist=allowlist)
        if not out:
            return "", 0.0
        out.sort(key=lambda r: r[2], reverse=True)
        return out[0][1], float(out[0][2])

    psnrs, ssims, laps, laps_norm = [], [], [], []
    confs, exact_matches, char_accs = [], [], []

    for fpath in files:
        fname = os.path.basename(fpath)
        plate_id = fname.rsplit("_v", 1)[0]
        gt_text = gt.get(plate_id, "")

        pred_bgr = cv2.imread(fpath)
        if pred_bgr is None:
            continue

        clear_path = os.path.join(clear_dir, fname)
        if os.path.isfile(clear_path):
            clear_bgr = cv2.imread(clear_path)
            if clear_bgr is not None and clear_bgr.shape == pred_bgr.shape:
                psnrs.append(psnr(clear_bgr, pred_bgr, data_range=255))
                ssims.append(ssim(clear_bgr, pred_bgr, channel_axis=2, data_range=255))

        laps.append(laplacian_var(pred_bgr))
        laps_norm.append(laplacian_var(pred_bgr, resize_to=NORM_SIZE))

        text, conf = ocr_top(fpath)
        confs.append(conf)
        if gt_text:
            exact_matches.append(1.0 if normalize_text(text) == normalize_text(gt_text) else 0.0)
            ca = char_accuracy(text, gt_text)
            if ca is not None:
                char_accs.append(ca)

    def avg(lst):
        return float(np.mean(lst)) if lst else float("nan")

    return {
        "n": len(files),
        "psnr": avg(psnrs),
        "ssim": avg(ssims),
        "laplacian_var": avg(laps),
        "laplacian_var_norm128": avg(laps_norm),
        "ocr_conf": avg(confs),
        "ocr_exact_match": avg(exact_matches),
        "ocr_char_acc": avg(char_accs),
    }


def print_row(cfg, r):
    print(f"{cfg:24s} n={r['n']:3d} PSNR={r['psnr']:6.2f} SSIM={r['ssim']:.4f} "
          f"LapNorm128={r['laplacian_var_norm128']:7.1f} OCRconf={r['ocr_conf']:.4f} "
          f"exact={r['ocr_exact_match']:.4f} charAcc={r['ocr_char_acc']:.4f}")


def main():
    compare_root = HERE
    gt = json.load(open(GT_PATH))

    configs = sorted(
        d for d in os.listdir(compare_root)
        if os.path.isdir(os.path.join(compare_root, d)) and not d.startswith("hazy_") and not d.startswith("clear_")
    )
    if not configs:
        print(f"No config output folders found under {compare_root}. Run generate_outputs.py first.")
        return

    print("Loading EasyOCR reader (GPU)...")
    reader = easyocr.Reader(["en"], gpu=True)
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    results = {}
    for cfg in configs:
        cfg_dir = os.path.join(compare_root, cfg)
        files = sorted(glob.glob(os.path.join(cfg_dir, "*.jpg")) + glob.glob(os.path.join(cfg_dir, "*.png")))
        if not files:
            continue
        sample = cv2.imread(files[0])
        res = sample.shape[0]
        clear_dir = os.path.join(compare_root, f"clear_{res}")
        results[cfg] = score_folder(files, clear_dir, gt, reader, allowlist)
        print_row(cfg, results[cfg])

    for res_dir in sorted(d for d in os.listdir(compare_root) if d.startswith("hazy_")):
        res = res_dir.split("_")[1]
        cfg = f"baseline_{res_dir}"
        files = sorted(glob.glob(os.path.join(compare_root, res_dir, "*.jpg")))
        clear_dir = os.path.join(compare_root, f"clear_{res}")
        results[cfg] = score_folder(files, clear_dir, gt, reader, allowlist)
        print_row(cfg, results[cfg])

    with open(os.path.join(compare_root, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(compare_root, "results.md"), "w") as f:
        f.write("# Mamba dehazing checkpoint comparison\n\n")
        f.write("| config | n | PSNR | SSIM | Laplacian var (native) | Laplacian var (norm128) | OCR conf | OCR exact match | OCR char acc |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for cfg, r in sorted(results.items()):
            f.write(f"| {cfg} | {r['n']} | {r['psnr']:.2f} | {r['ssim']:.4f} | {r['laplacian_var']:.1f} | "
                     f"{r['laplacian_var_norm128']:.1f} | {r['ocr_conf']:.4f} | {r['ocr_exact_match']:.4f} | "
                     f"{r['ocr_char_acc']:.4f} |\n")

    print(f"\nWrote {os.path.join(compare_root, 'results.json')} and results.md")


if __name__ == "__main__":
    main()
