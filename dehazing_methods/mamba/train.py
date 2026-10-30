"""
train.py
--------
Trains MambaDehazeNet on PAIRED (hazy, clear) images using the
supervised Charbonnier + exposure loss in losses.py.

Usage:
    python train.py --hazy_dir datasets/dataset/hazy --clear_dir datasets/dataset/clear --epochs 50

Changed from the zero-reference version:
  - Dataset is now PairedHazeDataset (needs both --hazy_dir and
    --clear_dir, not a single --foggy_dir) — supervision requires the
    real clean image, not just priors on the hazy image alone.
  - Loss call site passes the clear ground-truth image, not the
    transmission map: criterion(output, clear) instead of criterion(output, t).
  - Training from scratch is strongly recommended over resuming from a
    checkpoint that already collapsed under the old loss.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MambaDehazeNet, estimate_atmospheric_light, recover_clear_image
from losses import DehazeLoss
from paired_dataset import PairedHazeDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hazy_dir", type=str, required=True, help="Folder of real/synthetic hazy crops")
    p.add_argument("--clear_dir", type=str, required=True, help="Folder of matching clean crops")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--limit", type=int, default=None,
                    help="Randomly subsample the dataset to this many pairs "
                         "(e.g. 250) for faster iteration, instead of using all of it")
    p.add_argument("--seed", type=int, default=0, help="Seed for --limit subsampling")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--embed_dim", type=int, default=32)
    p.add_argument("--t_min", type=float, default=0.1)
    p.add_argument("--w_charbonnier", type=float, default=1.0)
    p.add_argument("--w_exposure", type=float, default=0.1)
    p.add_argument("--out_dir", type=str, default="./checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    dataset = PairedHazeDataset(args.hazy_dir, args.clear_dir, resolution=args.resolution,
                                 limit=args.limit, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, drop_last=True)

    model = MambaDehazeNet(embed_dim=args.embed_dim, t_min=args.t_min).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = DehazeLoss(w_charbonnier=args.w_charbonnier, w_exposure=args.w_exposure)

    print(f"Training on {len(dataset)} paired hazy/clear crops | device={device}")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running = {}
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)

            A = estimate_atmospheric_light(hazy)          # classical, no grad
            t = model(hazy)                                 # learned transmission map
            output = recover_clear_image(hazy, t, A, t_min=args.t_min)

            loss, logs = criterion(output, clear)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        n_batches = len(loader)
        avg = {k: v / n_batches for k, v in running.items()}
        print(f"[epoch {epoch+1}/{args.epochs}] "
              + " ".join(f"{k}={v:.4f}" for k, v in avg.items())
              + f" lr={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.out_dir, f"mamba_dehaze_epoch{epoch+1}.pt")
            torch.save({"model_state": model.state_dict(),
                        "embed_dim": args.embed_dim,
                        "t_min": args.t_min}, ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
