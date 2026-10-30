"""
diagnose_crops.py
------------------
Drop this next to your model.py / mamba_blocks.py / inference.py. It's a
targeted test to distinguish the two remaining hypotheses:

  (A) resolution/domain mismatch — model is fine, it just never saw
      full driving-scene photos at full resolution during training.
  (B) training collapse — the unsupervised losses converged to a
      degenerate solution and the model doesn't work even in-distribution.

Unlike inference.py, this also dumps the predicted transmission map `t`
and the estimated atmospheric light `A` for each crop, and prints their
stats — a collapsed/degenerate model usually shows up as `t` being
nearly constant (barely varying pixel-to-pixel) across every image,
regardless of haze content.

Usage:
    python diagnose_crops.py --checkpoint checkpoints/mamba_dehaze_epoch50.pt \
        --data_dir /path/to/real/plate_crops --out_dir ./diagnosis --limit 12

`--data_dir` should point at REAL plate crops (ideally some visibly
hazy/foggy ones), not the full road-scene photo — that's the point of
this test. Each crop is resized to 64x64 (your training resolution)
before inference, matching what the model actually learned on.
"""

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

from model import MambaDehazeNet, estimate_atmospheric_light, recover_clear_image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MambaDehazeNet(
        embed_dim=ckpt.get("embed_dim", 32), t_min=ckpt.get("t_min", 0.1)
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("t_min", 0.1)


def to_uint8(arr01):
    return (arr01.clip(0, 1) * 255).astype(np.uint8)


def make_triptych(hazy_u8, t_u8_gray, dehazed_u8):
    """Side-by-side: hazy | transmission map (grayscale, tiled to 3ch) | dehazed."""
    t_rgb = np.stack([t_u8_gray] * 3, axis=-1)
    h = max(hazy_u8.shape[0], t_rgb.shape[0], dehazed_u8.shape[0])
    pad = lambda im: np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)))
    strip = np.concatenate([pad(hazy_u8), pad(t_rgb), pad(dehazed_u8)], axis=1)
    return Image.fromarray(strip)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True, help="Folder of REAL plate crops")
    p.add_argument("--out_dir", default="./diagnosis")
    p.add_argument("--resolution", type=int, default=64, help="Should match training res")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    model, t_min = load_model(args.checkpoint, device)

    paths = sorted(
        p2 for ext in IMG_EXTS for p2 in glob.glob(os.path.join(args.data_dir, f"*{ext}"))
    )[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No images found in {args.data_dir}")

    print(f"{'file':30s}  {'t_min':>7s} {'t_mean':>7s} {'t_max':>7s} {'t_std':>7s}   "
          f"{'A_r':>5s} {'A_g':>5s} {'A_b':>5s}")

    all_t_means = []
    for path in paths:
        img = Image.open(path).convert("RGB").resize(
            (args.resolution, args.resolution), Image.BILINEAR
        )
        arr = np.array(img).astype(np.float32) / 255.0  # HxWx3 in [0,1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        A = estimate_atmospheric_light(tensor)
        t = model(tensor)
        output = recover_clear_image(tensor, t, A, t_min=t_min)

        t_np = t.squeeze().cpu().numpy()
        all_t_means.append(t_np.mean())

        fname = os.path.basename(path)
        print(f"{fname:30s}  {t_np.min():7.3f} {t_np.mean():7.3f} {t_np.max():7.3f} "
              f"{t_np.std():7.3f}   {A[0,0,0,0].item():5.2f} {A[0,1,0,0].item():5.2f} "
              f"{A[0,2,0,0].item():5.2f}")

        hazy_u8 = to_uint8(arr)
        t_gray_u8 = to_uint8(np.stack([t_np] * 1, axis=-1)).squeeze(-1)
        out_u8 = to_uint8(output.squeeze(0).permute(1, 2, 0).cpu().numpy())

        triptych = make_triptych(hazy_u8, t_gray_u8, out_u8)
        triptych.save(os.path.join(args.out_dir, f"cmp_{fname}"))

    all_t_means = np.array(all_t_means)
    print()
    print(f"across {len(paths)} crops: t_mean ranges {all_t_means.min():.3f}"
          f" .. {all_t_means.max():.3f} (std of per-image means: {all_t_means.std():.4f})")
    if all_t_means.std() < 0.01:
        print("  -> WARNING: transmission map barely varies image-to-image. "
              "That smells like training collapse (B), not a resolution issue: "
              "the network may have learned to predict roughly the same t "
              "regardless of input, which the recovery formula then always "
              "distorts the same way. Check the loss logs from training.")
    else:
        print("  -> t varies meaningfully across crops (haze-content-dependent), "
              "which is a good sign the network learned *something* real. If "
              "these dehazed outputs (see ./diagnosis/cmp_*.png) look reasonable "
              "while the full road-scene photo didn't, that confirms it's the "
              "resolution/domain mismatch (A), not training collapse.")

    print(f"\nsaved {len(paths)} hazy|transmission|dehazed comparison images to {args.out_dir}/")


if __name__ == "__main__":
    main()
