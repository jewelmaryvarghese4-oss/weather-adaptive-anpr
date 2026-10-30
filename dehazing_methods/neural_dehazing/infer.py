"""
infer.py

Run a trained dehazing model on a folder of hazy images.

Usage:
    python infer.py --checkpoint checkpoints/best.pt --input_dir hazy_test/ \
        --output_dir dehazed_out/
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from model import TransformerDehazeNet


def load_image(path, device):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # pad to multiple of 16 so up/downsampling stays aligned
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    t = torch.from_numpy(img_padded).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return t.to(device), h, w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})

    model = TransformerDehazeNet(
        base_ch=saved_args.get("base_ch", 32),
        bottleneck_blocks=saved_args.get("bottleneck_blocks", 4),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in exts]
    print(f"Running inference on {len(images)} images...")

    with torch.no_grad():
        for img_path in images:
            t, h, w = load_image(img_path, device)
            pred = model(t)
            pred = pred[:, :, :h, :w]  # crop back to original size
            pred_img = (pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / img_path.name), pred_img)

    print(f"Done. Dehazed images written to {out_dir}")


if __name__ == "__main__":
    main()
