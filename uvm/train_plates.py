"""
Fine-tunes UVMNet on datasets/dataset (ANPR plate/vehicle crops) --
250 clear scenes, ~8 hazy variants each. Starts from the Cityscapes-traffic
checkpoint (checkpoints_traffic/uvmnet_traffic_best_stateonly.pt) rather
than training from scratch: 250 scenes is small for a ~116M-param model,
and low-level haze-removal features should transfer even though the
content domain (close-up crops vs. street scenes) is very different --
see the conversation's diagnosis of the green-cast/darkening failures on
this exact data with the traffic-only checkpoint.

No train/val split exists in the raw dataset, so this carves one out by
scene id (not by individual hazy variant, to avoid leaking near-duplicate
haze variants of the same scene across the split).

Run from WSL2 (needs the real mamba_ssm CUDA kernel):
    source ~/mamba-venv/bin/activate
    cd /mnt/c/Users/Jewel/OneDrive/Desktop/jewwww/uvm
    python train_plates.py
"""
import argparse
import os
import random
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_uvmnet import UVMNet
from dataset_plates import PairedPlateDataset
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
    ap.add_argument("--data_root", default="../datasets/dataset")
    ap.add_argument("--resize", type=int, default=128)
    ap.add_argument("--val_frac", type=float, default=0.1,
                     help="Fraction of the 250 scenes held out for validation.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5,
                     help="Low LR for fine-tuning from a converged checkpoint, "
                          "not the 2e-4 used for from-scratch training.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default="cuda", choices=["torch", "cuda"])
    ap.add_argument("--uvmb_size", type=int, default=32)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--init_ckpt", default="checkpoints_traffic/uvmnet_traffic_best_stateonly.pt",
                     help="Raw state_dict checkpoint to fine-tune from. Empty string trains from scratch.")
    ap.add_argument("--ckpt_dir", default="checkpoints_plates")
    ap.add_argument("--ckpt_every", type=int, default=5)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--lambda_grad", type=float, default=0.15,
                     help="Weight on the Sobel gradient loss (detail/sharpness preservation).")
    ap.add_argument("--lambda_reg", type=float, default=0.02,
                     help="Weight on the residual-magnitude regularizer (discourages large "
                          "corrections, aimed at the brightness-collapse problem on OOD input).")
    ap.add_argument("--density_aug", action="store_true", default=True,
                     help="Physically-consistent haze-density augmentation (see haze_aug.py): "
                          "alpha-blends clear/hazy per sample so training sees a continuous "
                          "density range instead of only the 8 fixed variants per scene, "
                          "including denser-than-observed and near-zero haze.")
    ap.add_argument("--no_density_aug", dest="density_aug", action="store_false")
    ap.add_argument("--density_alpha_min", type=float, default=0.0)
    ap.add_argument("--density_alpha_max", type=float, default=1.3)
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    all_scenes = PairedPlateDataset.all_scene_ids(args.data_root)
    rng = random.Random(0)
    rng.shuffle(all_scenes)
    n_val = max(1, int(len(all_scenes) * args.val_frac))
    val_scenes = set(all_scenes[:n_val])
    train_scenes = set(all_scenes[n_val:])
    print(f"scenes: {len(all_scenes)} total -> {len(train_scenes)} train / {len(val_scenes)} val")

    train_set = PairedPlateDataset(args.data_root, resize=args.resize, augment=True, scene_ids=train_scenes,
                                    density_aug=args.density_aug,
                                    density_alpha_range=(args.density_alpha_min, args.density_alpha_max))
    val_set = PairedPlateDataset(args.data_root, resize=args.resize, augment=False, scene_ids=val_scenes)
    print(f"train pairs: {len(train_set)} | val pairs: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
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
            path = os.path.join(args.ckpt_dir, f"uvmnet_plates_epoch{epoch}.pt")
            torch.save(state, path)
            print(f"  saved -> {path}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state["best_val_loss"] = best_val_loss
            torch.save(state, os.path.join(args.ckpt_dir, "uvmnet_plates_best.pt"))

    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "uvmnet_plates_final.pt"))
    print(f"done -> {os.path.join(args.ckpt_dir, 'uvmnet_plates_final.pt')}")


if __name__ == "__main__":
    main()
