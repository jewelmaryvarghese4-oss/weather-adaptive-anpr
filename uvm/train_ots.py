"""
Fine-tunes UVMNet on RESIDE-beta OTS (Outdoor Training Set) -- real-photo
outdoor scenes with synthetic haze, paired against clear ground truth.
Uses the same CombinedDehazeLoss recipe as train_plates.py (Charbonnier +
Sobel-gradient + residual-magnitude regularizer), not the plain Charbonnier
train_traffic.py uses, since that's the more recent/tuned loss.

Starts from the traffic checkpoint by default (checkpoints_traffic/
uvmnet_traffic_best_stateonly.pt) -- OTS is general outdoor photography,
closer in domain to the Cityscapes-traffic scenes (full outdoor frames)
than to the close-up plate/vehicle crops. Override --init_ckpt to start
from the plates checkpoint instead if that's the deployment target.

No train/val split exists in the raw download, so this carves one out by
clear-image id (not by individual haze variant), same reasoning as
train_plates.py: avoids leaking near-duplicate haze variants of the same
scene across the split.

Run from WSL2 (needs the real mamba_ssm CUDA kernel):
    source ~/mamba-venv/bin/activate
    cd /mnt/c/Users/Jewel/OneDrive/Desktop/jewwww/uvm
    python train_ots.py --data_root ../datasets/OTS
"""
import argparse
import os
import random
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_uvmnet import UVMNet
from dataset_ots import PairedOTSDataset
from losses import CombinedDehazeLoss


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
            loss, _parts = criterion(pred, clear, hazy)
        pred_clamped = torch.clamp(pred.float(), 0.0, 1.0)
        total_loss += loss.item() * hazy.size(0)
        total_psnr += psnr(pred_clamped, clear).item() * hazy.size(0)
        n += hazy.size(0)
    model.train()
    return total_loss / n, total_psnr / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../datasets/OTS",
                     help="Must contain <data_root>/clear/*.jpg and <data_root>/haze/*.jpg "
                          "exactly as the OTS_BETA Box download extracts.")
    ap.add_argument("--patch_size", type=int, default=256,
                     help="Synchronized random crop size. OTS images are full outdoor "
                          "photos of varying resolution, not fixed-size like Cityscapes.")
    ap.add_argument("--val_frac", type=float, default=0.05,
                     help="Fraction of clear-image ids held out for validation.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--val_batch_size", type=int, default=8)
    ap.add_argument("--val_limit", type=int, default=300,
                     help="Subsample val pairs for a fast per-epoch check.")
    ap.add_argument("--limit", type=int, default=None,
                     help="Subsample train pairs, e.g. for a quick smoke test.")
    ap.add_argument("--lr", type=float, default=5e-5,
                     help="Low LR for fine-tuning from a converged checkpoint.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default="cuda", choices=["torch", "cuda"])
    ap.add_argument("--uvmb_size", type=int, default=32)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--init_ckpt", default="checkpoints_traffic/uvmnet_traffic_best_stateonly.pt",
                     help="Raw state_dict checkpoint to fine-tune from. Empty string trains from scratch.")
    ap.add_argument("--ckpt_dir", default="checkpoints_ots")
    ap.add_argument("--ckpt_every", type=int, default=2)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--lambda_grad", type=float, default=0.15)
    ap.add_argument("--lambda_reg", type=float, default=0.02)
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    all_ids = PairedOTSDataset.all_clear_ids(args.data_root)
    rng = random.Random(0)
    rng.shuffle(all_ids)
    n_val = max(1, int(len(all_ids) * args.val_frac))
    val_ids = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])
    print(f"clear ids: {len(all_ids)} total -> {len(train_ids)} train / {len(val_ids)} val")

    train_set = PairedOTSDataset(args.data_root, patch_size=args.patch_size, augment=True,
                                  clear_ids=train_ids, limit=args.limit)
    val_set = PairedOTSDataset(args.data_root, patch_size=args.patch_size, augment=False,
                                clear_ids=val_ids, limit=args.val_limit)
    print(f"train pairs: {len(train_set)} | val pairs: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = UVMNet(n_channels=3, backend=args.backend, uvmb_size=args.uvmb_size).to(args.device)
    if args.init_ckpt:
        sd = torch.load(args.init_ckpt, map_location=args.device)
        model.load_state_dict(sd)
        print(f"initialized from {args.init_ckpt}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device.startswith("cuda"))
    criterion = CombinedDehazeLoss(lambda_grad=args.lambda_grad, lambda_reg=args.lambda_reg).to(args.device)

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
        running_parts = {"charbonnier": 0.0, "grad": 0.0, "reg": 0.0}
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch:3d}/{args.epochs}")
        for hazy, clear in pbar:
            hazy, clear = hazy.to(args.device), clear.to(args.device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu",
                                 dtype=torch.float16, enabled=scaler.is_enabled()):
                pred = model(hazy)
                loss, parts = criterion(pred, clear, hazy)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            for k in running_parts:
                running_parts[k] += parts[k].item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", pix=f"{parts['charbonnier'].item():.4f}")
        sched.step()

        train_loss = running / len(train_loader)
        val_loss, val_psnr = validate(model, val_loader, criterion, args.device, scaler.is_enabled())
        elapsed = time.time() - t0
        n_batches = len(train_loader)
        print(f"epoch {epoch:3d} | train {train_loss:.4f} "
              f"(pix {running_parts['charbonnier']/n_batches:.4f}, "
              f"grad {running_parts['grad']/n_batches:.4f}, "
              f"reg {running_parts['reg']/n_batches:.4f}) "
              f"| val {val_loss:.4f} | val_psnr {val_psnr:.2f}dB "
              f"| lr {sched.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            path = os.path.join(args.ckpt_dir, f"uvmnet_ots_epoch{epoch}.pt")
            torch.save(state, path)
            print(f"  saved -> {path}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state["best_val_loss"] = best_val_loss
            torch.save(state, os.path.join(args.ckpt_dir, "uvmnet_ots_best.pt"))

    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "uvmnet_ots_final.pt"))
    print(f"done -> {os.path.join(args.ckpt_dir, 'uvmnet_ots_final.pt')}")


if __name__ == "__main__":
    main()
