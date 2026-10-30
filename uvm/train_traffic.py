"""
Trains UVMNet on the UVM_TRAFFIC_DATASET (Foggy Cityscapes) paired
clear/foggy scenes, supervised Charbonnier loss (same recipe as
train_uvmnet.py -- see that file's docstring for why: direct residual
regression against real clean images, no zero-reference/dark-channel
priors).

Run from WSL2 with the mamba_ssm-enabled venv (needed for --backend cuda,
i.e. the real fused selective-scan kernel):

    source ~/mamba-venv/bin/activate
    cd /mnt/c/Users/Jewel/OneDrive/Desktop/jewwww/uvm
    python train_traffic.py

Defaults (uvmb_size=32, patch=512, batch=4) were picked by probing actual
VRAM use on this 8GB RTX 4060: ~6.24GB peak with AMP on, leaving headroom
for DataLoader workers and cuDNN workspace. patch=512 (vs. the smaller
384 crop originally used for ANPR-crop training) was chosen as a middle
ground to narrow the train/deploy scale gap without tripling epoch time --
UVMB always resizes to a fixed uvmb_size x uvmb_size grid internally, so
training on small crops teaches it a much higher pixels-per-grid-cell
ratio than a full 2048x1024 deployment frame actually has. uvmb_size=64
(the paper default, ~695M params) reliably OOMs on 8GB even at small
patch sizes -- don't raise it without re-probing.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_uvmnet import UVMNet
from dataset_traffic import PairedTrafficDataset


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


@torch.no_grad()
def psnr(pred, target):
    mse = torch.clamp((pred - target) ** 2, min=1e-10).mean()
    return -10.0 * torch.log10(mse)


@torch.no_grad()
def validate(model, loader, criterion, device, amp):
    model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    for hazy, clear in loader:
        hazy, clear = hazy.to(device), clear.to(device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                             dtype=torch.float16, enabled=amp):
            pred = model(hazy)
            loss = criterion(pred, clear)
        pred_clamped = torch.clamp(pred.float(), 0.0, 1.0)
        total_loss += loss.item() * hazy.size(0)
        total_psnr += psnr(pred_clamped, clear).item() * hazy.size(0)
        n += hazy.size(0)
    model.train()
    return total_loss / n, total_psnr / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear_dir", default="../UVM_TRAFFIC_DATASET/leftImg8bit/train")
    ap.add_argument("--hazy_dir", default="../UVM_TRAFFIC_DATASET/leftImg8bit_foggy/train")
    ap.add_argument("--val_clear_dir", default="../UVM_TRAFFIC_DATASET/leftImg8bit/val")
    ap.add_argument("--val_hazy_dir", default="../UVM_TRAFFIC_DATASET/leftImg8bit_foggy/val")
    ap.add_argument("--patch_size", type=int, default=512,
                     help="512 (batch=4) probed at ~6.24GB peak on an 8GB card -- the "
                          "chosen middle ground between training speed and the "
                          "train/deploy scale gap: UVMB always resizes to a fixed "
                          "uvmb_size x uvmb_size grid, so a small training patch teaches "
                          "it to expect far more spatial downsampling per grid cell than "
                          "a full 2048x1024 deployment frame actually needs. 768 narrows "
                          "that gap further but roughly triples epoch time; 384 is faster "
                          "but furthest from deployment scale.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--val_batch_size", type=int, default=4)
    ap.add_argument("--val_limit", type=int, default=200,
                     help="Subsample val pairs for a fast per-epoch check "
                          "(full val is 1500 pairs at 3 betas/scene).")
    ap.add_argument("--limit", type=int, default=None,
                     help="Subsample train pairs, e.g. for a quick smoke test.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default="cuda", choices=["torch", "cuda"],
                     help="'cuda' needs the real mamba_ssm kernel (WSL2 venv). "
                          "'torch' backend (uvmb.py) is not implemented in this repo yet.")
    ap.add_argument("--uvmb_size", type=int, default=32,
                     help="Internal SSM resize target. 32 probed at ~4.4GB peak "
                          "(patch=384, batch=4, AMP on) on an 8GB card. 64 (paper "
                          "default) OOMs on 8GB -- re-probe before raising this.")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--ckpt_dir", default="checkpoints_traffic")
    ap.add_argument("--ckpt_every", type=int, default=5)
    ap.add_argument("--resume", default=None, help="Path to a checkpoint to resume from.")
    args = ap.parse_args()

    if args.device == "cpu":
        print("WARNING: training UVMNet on CPU will be extremely slow. "
              "Use a CUDA device (see this file's docstring for the WSL2 setup).")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_set = PairedTrafficDataset(args.clear_dir, args.hazy_dir,
                                      patch_size=args.patch_size, augment=True,
                                      limit=args.limit)
    val_set = PairedTrafficDataset(args.val_clear_dir, args.val_hazy_dir,
                                    patch_size=args.patch_size, augment=False,
                                    limit=args.val_limit)
    print(f"train pairs: {len(train_set)} | val pairs: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = UVMNet(n_channels=3, backend=args.backend, uvmb_size=args.uvmb_size).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device.startswith("cuda"))
    charbonnier = CharbonnierLoss()

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch:3d}/{args.epochs}")
        for hazy, clear in pbar:
            hazy, clear = hazy.to(args.device), clear.to(args.device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu",
                                 dtype=torch.float16, enabled=scaler.is_enabled()):
                pred = model(hazy)
                loss = charbonnier(pred, clear)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            pbar.set_postfix(charbonnier=f"{loss.item():.4f}")
        sched.step()

        train_loss = running / len(train_loader)
        val_loss, val_psnr = validate(model, val_loader, charbonnier, args.device, scaler.is_enabled())
        elapsed = time.time() - t0
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| val_psnr {val_psnr:.2f}dB | lr {sched.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            path = os.path.join(args.ckpt_dir, f"uvmnet_traffic_epoch{epoch}.pt")
            torch.save(state, path)
            print(f"  saved -> {path}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state["best_val_loss"] = best_val_loss
            torch.save(state, os.path.join(args.ckpt_dir, "uvmnet_traffic_best.pt"))

    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "uvmnet_traffic_final.pt"))
    print(f"done -> {os.path.join(args.ckpt_dir, 'uvmnet_traffic_final.pt')}")


if __name__ == "__main__":
    main()
