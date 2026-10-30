"""
fog_synthesis.py

Generates synthetic paired (hazy, clear) training data from a folder of
clear number-plate / scene images, using the standard atmospheric
scattering model:

    I(x) = J(x) * t(x) + A * (1 - t(x))

where:
    J(x) = clear scene radiance (your original image)
    A    = atmospheric light (usually near-white, slightly biased by scene)
    t(x) = transmission map = exp(-beta * d(x))
    d(x) = scene "depth" (for close-up plate shots this is roughly
           uniform, so we synthesize a smooth pseudo-depth field to
           vary fog density realistically across the plate/vehicle area)
    beta = fog/haze density coefficient (higher = thicker fog)

Usage:
    python fog_synthesis.py --input_dir clear_images/ --output_dir ../datasets/dataset/ \
        --variants 4 --beta_range 0.5 2.5

Produces:
    ../datasets/dataset/clear/<name>.jpg          (copy of original)
    ../datasets/dataset/hazy/<name>_v0.jpg ...    (N hazy variants per image)
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np


def synthetic_depth_map(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build a smooth pseudo-depth map in [0, 1] using low-frequency random
    noise (cheap Perlin-like effect via upsampled Gaussian-blurred noise).
    This avoids needing a real depth estimator, which is unnecessary for
    close-range plate/vehicle crops where fog density varies gently.
    """
    small_h, small_w = max(4, h // 32), max(4, w // 32)
    noise = rng.random((small_h, small_w)).astype(np.float32)
    depth = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=w / 8)
    # Normalize to [0, 1]
    depth -= depth.min()
    depth /= (depth.max() + 1e-6)
    return depth


def add_fog(
    img: np.ndarray,
    beta: float,
    atmos_light: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply the atmospheric scattering model to a single BGR uint8 image.
    """
    img_f = img.astype(np.float32) / 255.0
    h, w = img_f.shape[:2]

    depth = synthetic_depth_map(h, w, rng)
    transmission = np.exp(-beta * depth)
    transmission = np.clip(transmission, 0.15, 1.0)  # avoid fully opaque fog
    transmission = transmission[..., None]  # broadcast over channels

    # Slight color bias on atmospheric light (real fog isn't pure white)
    A = np.array(
        [
            atmos_light + rng.uniform(-0.03, 0.03),
            atmos_light + rng.uniform(-0.03, 0.03),
            atmos_light + rng.uniform(-0.03, 0.03),
        ],
        dtype=np.float32,
    )

    hazy = img_f * transmission + A * (1 - transmission)

    # Mild sensor noise, since real foggy captures are noisier
    noise_sigma = rng.uniform(0.002, 0.01)
    hazy += rng.normal(0, noise_sigma, hazy.shape).astype(np.float32)

    hazy = np.clip(hazy, 0, 1)
    return (hazy * 255).astype(np.uint8)


def process_folder(
    input_dir: str,
    output_dir: str,
    variants: int,
    beta_range: tuple,
    atmos_range: tuple,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)

    input_path = Path(input_dir)
    clear_out = Path(output_dir) / "clear"
    hazy_out = Path(output_dir) / "hazy"
    clear_out.mkdir(parents=True, exist_ok=True)
    hazy_out.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in input_path.iterdir() if p.suffix.lower() in exts]

    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}")

    print(f"Found {len(images)} clear images. Generating {variants} hazy variants each...")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  skipping unreadable file: {img_path}")
            continue

        name = img_path.stem
        cv2.imwrite(str(clear_out / f"{name}.jpg"), img)

        for v in range(variants):
            beta = rng.uniform(*beta_range)
            atmos = rng.uniform(*atmos_range)
            hazy_img = add_fog(img, beta, atmos, rng)
            cv2.imwrite(str(hazy_out / f"{name}_v{v}.jpg"), hazy_img)

    print(f"Done. Clear images in {clear_out}, hazy images in {hazy_out}.")
    print("Pairing convention for the Dataset loader: 'name_v0.jpg' (hazy) <-> 'name.jpg' (clear).")


def main():
    parser = argparse.ArgumentParser(description="Synthesize hazy/foggy training pairs.")
    parser.add_argument("--input_dir", required=True, help="Folder of clear images")
    parser.add_argument("--output_dir", required=True, help="Where to write clear/ and hazy/ subfolders")
    parser.add_argument("--variants", type=int, default=4, help="Hazy variants per clear image")
    parser.add_argument("--beta_range", type=float, nargs=2, default=[0.5, 2.5],
                         help="Min/max fog density coefficient")
    parser.add_argument("--atmos_range", type=float, nargs=2, default=[0.7, 0.95],
                         help="Min/max atmospheric light brightness (0-1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    process_folder(
        args.input_dir,
        args.output_dir,
        args.variants,
        tuple(args.beta_range),
        tuple(args.atmos_range),
        args.seed,
    )


if __name__ == "__main__":
    main()
