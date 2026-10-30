"""
train.py

Trains the dehazing model on paired synthetic hazy/clear data.

Usage:
    python train.py --data_dir ../datasets/dataset/ --epochs 50 \
        --batch_size 8 --patch_size 256 --lr 2e-4

Splits the pairs 90/10 into train/val automatically.
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import FogPairDataset
from losses import DehazeLoss
from model import TransformerDehazeNet


def psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 99.0
    return 10 * math.log10(1.0 / mse)


def evaluate(model, loader, device):
    model.eval()
    total_psnr = 0.0
    n = 0
    with torch.no_grad():
        for hazy, clear in loader:
            hazy, clear = hazy.to(device), clear.to(device)
            pred = model(hazy)
            total_psnr += psnr(pred, clear) * hazy.size(0)
            n += hazy.size(0)
    model.train()
    return total_psnr / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--bottleneck_blocks", type=int, default=4)
    parser.add_argument("--no_perceptual", action="store_true",
                         help="Disable VGG perceptual loss (faster, less sharp)")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    full_dataset = FogPairDataset(args.data_dir, patch_size=args.patch_size, train=True)
    val_len = max(1, int(0.1 * len(full_dataset)))
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = TransformerDehazeNet(base_ch=args.base_ch,
                                  bottleneck_blocks=args.bottleneck_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TransformerDehazeNet, params: {n_params / 1e6:.2f}M")

    criterion = DehazeLoss(use_perceptual=not args.no_perceptual).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_psnr = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for i, (hazy, clear) in enumerate(train_loader):
            hazy, clear = hazy.to(device), clear.to(device)

            pred = model(hazy)
            loss, logs = criterion(pred, clear)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += logs["total"]

            if i % 20 == 0:
                print(f"epoch {epoch} step {i}/{len(train_loader)} "
                      f"loss {logs['total']:.4f} (l1 {logs['l1']:.4f}, "
                      f"ssim {logs['ssim_loss']:.4f})")

        scheduler.step()
        val_psnr = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start
        print(f"== epoch {epoch} done in {elapsed:.1f}s | "
              f"avg train loss {running_loss / len(train_loader):.4f} | "
              f"val PSNR {val_psnr:.2f} dB ==")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "last.pt"))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            ckpt["best_psnr"] = best_psnr
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  new best model saved (PSNR {best_psnr:.2f} dB)")

    print(f"Training complete. Best val PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
