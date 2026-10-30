"""
The real end-to-end test: dehaze (dispatch_dehaze.py's UVMNet/CLAHE
routing) THEN read the plate (LPRNet) -- something never actually tested
together before now. LPRNet's training/validation so far only ever saw
raw hazy crops directly; this checks whether dehazing first actually
helps OCR accuracy, hurts it, or does nothing.

Uses the 21 held-out scenes from train_lprnet.py's val split (same
val_frac=0.1, seed=0 -- reproduced here, not re-randomized) -- LPRNet's
weights never saw these scenes' ground truth text during training, so
this is a genuine unseen-data test, not just re-scoring the training set.

Three conditions per hazy variant, all read by the same LPRNet checkpoint:
    raw_hazy  -- straight into LPRNet, no dehazing at all
    dehazed   -- dispatch_dehaze.py's routing (contrast-based UVMNet/CLAHE
                 pick) first, then LPRNet
    clear_gt  -- the actual clear ground-truth image straight into LPRNet
                 (once per scene, not per variant) -- the ceiling: how
                 well LPRNet reads a plate with NO haze at all

--skip_dehaze skips the UVMNet path entirely (no loading it, no "dehazed"
condition/column) -- just raw_hazy vs clear_gt, for when dehazing is out
of scope and you only want to evaluate detection/OCR.

Run from WSL2 (needs the real mamba_ssm CUDA kernel for the UVMNet path,
unless --skip_dehaze is set):
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python evaluate_pipeline_ocr.py
    python evaluate_pipeline_ocr.py --skip_dehaze
"""
import argparse
import csv
import json
import os
import random

import torch
from PIL import Image
from torchvision import transforms

import test_uvmnet as uvm
from model_uvmnet import UVMNet
from model_lprnet import LPRNet
from clahe_preprocess import apply_clahe_pil
from dispatch_dehaze import local_contrast, DEFAULT_CONTRAST_THRESHOLD
from dataset_lprnet import INPUT_SIZE as LPRNET_INPUT_SIZE
from lprnet_vocab import NUM_CLASSES, ctc_greedy_decode
from two_line_plate import normalize_plate_layout
from train_lprnet import levenshtein


def get_held_out_scene_ids(ground_truth_path, val_frac=0.1, seed=0):
    with open(ground_truth_path) as f:
        gt = json.load(f)
    scene_ids = list(gt.keys())
    rng = random.Random(seed)
    rng.shuffle(scene_ids)
    n_val = max(1, int(len(scene_ids) * val_frac))
    return gt, set(scene_ids[:n_val])


@torch.no_grad()
def read_plate(lprnet, image: Image.Image, device) -> str:
    img = normalize_plate_layout(image.convert("RGB")).resize(LPRNET_INPUT_SIZE)
    img_t = transforms.ToTensor()(img).unsqueeze(0).to(device)
    log_probs = torch.log_softmax(lprnet(img_t), dim=1).permute(2, 0, 1)
    return ctc_greedy_decode(log_probs.cpu())[0]


def dehaze(uvmnet, image: Image.Image, device) -> Image.Image:
    """Same routing logic as dispatch_dehaze.py: low local contrast ->
    CLAHE, otherwise -> UVMNet."""
    c = local_contrast(image)
    if c < DEFAULT_CONTRAST_THRESHOLD:
        return apply_clahe_pil(image)
    return uvm.process_image(uvmnet, image, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../datasets/dataset")
    ap.add_argument("--ground_truth", default="../evaluation/baseline_ocr/ground_truth.json")
    ap.add_argument("--uvmnet_ckpt", default="checkpoints_combined/uvmnet_combined_best_stateonly.pt")
    ap.add_argument("--lprnet_ckpt", default="checkpoints_lprnet_mixed/lprnet_best.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default="cuda", choices=["torch", "cuda"])
    ap.add_argument("--uvmb_size", type=int, default=32)
    ap.add_argument("--out_csv", default="pipeline_ocr_results.csv")
    ap.add_argument("--skip_dehaze", action="store_true",
                     help="Skip UVMNet entirely -- no loading it, no 'dehazed' "
                          "condition -- just raw_hazy vs clear_gt.")
    args = ap.parse_args()

    device = torch.device(args.device)

    uvmnet = None
    if not args.skip_dehaze:
        print("Loading UVMNet...")
        uvmnet = UVMNet(n_channels=3, backend=args.backend, uvmb_size=args.uvmb_size).to(device)
        uvmnet.load_state_dict(torch.load(args.uvmnet_ckpt, map_location=device))
        uvmnet.eval()

    print("Loading LPRNet...")
    lprnet = LPRNet(num_classes=NUM_CLASSES).to(device)
    lprnet.load_state_dict(torch.load(args.lprnet_ckpt, map_location=device))
    lprnet.eval()

    gt, held_out_ids = get_held_out_scene_ids(args.ground_truth)
    print(f"{len(held_out_ids)} held-out scenes\n")

    clear_dir = os.path.join(args.data_root, "clear")
    hazy_dir = os.path.join(args.data_root, "hazy")
    hazy_files = sorted(os.listdir(hazy_dir))

    conditions = ("raw_hazy", "clear_gt") if args.skip_dehaze else ("raw_hazy", "dehazed", "clear_gt")
    rows = []
    metrics = {k: {"exact": 0, "cer_num": 0, "cer_den": 0, "n": 0} for k in conditions}

    for scene_id in sorted(held_out_ids):
        text = gt[scene_id]

        clear_path = os.path.join(clear_dir, scene_id + ".jpg")
        if os.path.exists(clear_path):
            clear_img = Image.open(clear_path).convert("RGB")
            pred = read_plate(lprnet, clear_img, device)
            m = metrics["clear_gt"]
            m["n"] += 1
            m["exact"] += int(pred == text)
            m["cer_num"] += levenshtein(pred, text)
            m["cer_den"] += max(len(text), 1)
            rows.append([scene_id, "clear_gt", text, pred, pred == text])
            print(f"[{scene_id}] clear_gt   gt={text!r:14s} pred={pred!r:14s} {'OK' if pred==text else ''}")

        variants = [f for f in hazy_files if f.startswith(scene_id + "_v")]
        for fname in variants:
            hazy_path = os.path.join(hazy_dir, fname)
            hazy_img = Image.open(hazy_path).convert("RGB")

            pred_raw = read_plate(lprnet, hazy_img, device)
            m = metrics["raw_hazy"]
            m["n"] += 1
            m["exact"] += int(pred_raw == text)
            m["cer_num"] += levenshtein(pred_raw, text)
            m["cer_den"] += max(len(text), 1)
            rows.append([scene_id, "raw_hazy", text, pred_raw, pred_raw == text])

            if args.skip_dehaze:
                print(f"[{scene_id}] {fname:24s} gt={text!r:14s} raw={pred_raw!r:14s}")
                continue

            dehazed_img = dehaze(uvmnet, hazy_img, device)
            pred_dehazed = read_plate(lprnet, dehazed_img, device)
            m = metrics["dehazed"]
            m["n"] += 1
            m["exact"] += int(pred_dehazed == text)
            m["cer_num"] += levenshtein(pred_dehazed, text)
            m["cer_den"] += max(len(text), 1)
            rows.append([scene_id, "dehazed", text, pred_dehazed, pred_dehazed == text])

            print(f"[{scene_id}] {fname:24s} gt={text!r:14s} raw={pred_raw!r:14s} dehazed={pred_dehazed!r:14s}")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "condition", "ground_truth", "prediction", "exact_match"])
        writer.writerows(rows)

    print("\n=== SUMMARY (held-out scenes only) ===")
    print(f"{'Condition':<12} {'N':>5} {'Exact acc':>10} {'CER':>8}")
    for cond in conditions:
        m = metrics[cond]
        acc = m["exact"] / max(m["n"], 1)
        cer = m["cer_num"] / max(m["cer_den"], 1)
        print(f"{cond:<12} {m['n']:>5} {acc:>9.2%} {cer:>8.4f}")
    print(f"\nFull per-image results: {args.out_csv}")


if __name__ == "__main__":
    main()
