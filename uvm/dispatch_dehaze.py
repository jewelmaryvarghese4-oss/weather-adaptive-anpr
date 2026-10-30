"""
Per-image dispatcher between UVMNet and classical CLAHE, chosen by a
local-contrast measure -- NOT a merge of the two at the pixel or training
level (both of those were tried and both hurt: CLAHE feeding into UVMNet
needed retraining that didn't converge well, and UVMNet output fed through
CLAHE just amplified whatever grain UVMNet already produced). This instead
runs each tool completely independently, untouched, and picks per-image
which finished output to keep.

Why contrast, not brightness: brightness doesn't predict which images
UVMNet struggles on -- nn.jpg (mean=98.4, the darkest test image) is one
UVMNet handles well, while heavyfog.jpg (mean=142.6, brighter than nn.jpg)
is the one that produces heavy grain. Grayscale stddev (local contrast)
separates them cleanly instead: heavyfog.jpg=20.65 vs. everything else in
this project's test set at 26+ (see the calibration this threshold was
picked from). Dense/uniform fog flattens local contrast regardless of how
dark the scene is, which is the actual condition UVMNet struggles with --
not darkness per se.

Routing:
    stddev >= --contrast_threshold  -> UVMNet   (already excellent here)
    stddev <  --contrast_threshold  -> CLAHE    (cleaner on hard/flat cases,
                                                  even though it can't
                                                  recover genuinely zero-
                                                  signal regions either)

Run from WSL2 (needs the real mamba_ssm CUDA kernel for the UVMNet path):
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python dispatch_dehaze.py \
        --input ../datasets/dataset2/test_samples \
        --checkpoint checkpoints_combined/uvmnet_combined_best_stateonly.pt \
        --output test_outputs_dispatch
"""
import argparse
import os

import torch
from PIL import Image, ImageStat

import test_uvmnet as uvm
from model_uvmnet import UVMNet
from clahe_preprocess import apply_clahe_pil

DEFAULT_CONTRAST_THRESHOLD = 24.0


def local_contrast(image: Image.Image) -> float:
    """Grayscale stddev -- see this file's docstring for why this (not
    brightness) is the routing signal."""
    return ImageStat.Stat(image.convert("L")).stddev[0]


def main():
    parser = argparse.ArgumentParser(description="Contrast-based dispatcher: UVMNet vs. CLAHE per image")
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--output", default="test_outputs_dispatch")
    parser.add_argument("--checkpoint", default="checkpoints_combined/uvmnet_combined_best_stateonly.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--uvmb_size", type=int, default=32)
    parser.add_argument("--tile_size", type=int, default=uvm.TILE_SIZE)
    parser.add_argument("--overlap", type=int, default=uvm.OVERLAP)
    parser.add_argument("--contrast_threshold", type=float, default=DEFAULT_CONTRAST_THRESHOLD,
                         help="Grayscale stddev cutoff. Below this -> CLAHE, at/above -> UVMNet. "
                              "Default calibrated against this project's test_samples set (see docstring).")
    args = parser.parse_args()

    uvm.TILE_SIZE = args.tile_size
    uvm.OVERLAP = args.overlap
    uvm.STRIDE = args.tile_size - args.overlap

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is not available.")
    device = torch.device(args.device)

    print("=" * 60)
    print("Contrast-based Dispatcher: UVMNet vs. CLAHE")
    print("=" * 60)
    print(f"Contrast threshold : {args.contrast_threshold} (stddev)")
    print(f"Checkpoint          : {args.checkpoint}")
    print()
    print("Loading UVMNet...")
    model = UVMNet(n_channels=3, backend=args.backend, uvmb_size=args.uvmb_size).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("Model loaded.")

    image_files = uvm.get_image_files(args.input)
    if not image_files:
        raise RuntimeError(f"No images found in: {args.input}")
    print()
    print(f"Found {len(image_files)} image(s).")
    os.makedirs(args.output, exist_ok=True)

    n_uvmnet, n_clahe = 0, 0
    for index, image_path in enumerate(image_files, start=1):
        name = os.path.basename(image_path)
        print()
        print(f"[{index}/{len(image_files)}] {name}")
        try:
            image = Image.open(image_path).convert("RGB")
            contrast = local_contrast(image)

            if contrast < args.contrast_threshold:
                route = "CLAHE"
                n_clahe += 1
                output_image = apply_clahe_pil(image)
            else:
                route = "UVMNet"
                n_uvmnet += 1
                output_image = uvm.process_image(model, image, device)

            print(f"    contrast(stddev)={contrast:.2f} -> routed to {route}")

            stem, _ = os.path.splitext(name)
            output_path = os.path.join(args.output, f"{stem}_restored.png")
            output_image.save(output_path)
            print(f"    Saved: {output_path}")

        except Exception as e:
            print(f"    ERROR: {e}")

        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print(f"Done. UVMNet: {n_uvmnet} image(s) | CLAHE: {n_clahe} image(s)")
    print(f"Results: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
