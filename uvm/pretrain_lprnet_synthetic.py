"""
Pretrains LPRNet on synthetic plates (synthetic_plate_dataset.py) before
fine-tuning on the ~197 real labeled plates in train_lprnet.py. See that
file's docstring and synthetic_plate_dataset.py's docstring for why this
two-stage approach exists: too few distinct real plate texts to learn
general character recognition from scratch.

--ccpd_root (optional): mixes in real photographs from CCPD (Chinese City
Parking Dataset, see dataset_ccpd.py's docstring) alongside the synthetic
plates. Synthetic data supplies text diversity but every pixel is
rendered; CCPD supplies real camera/lens/lighting/blur variety that
rendering can't -- even though its plate FORMAT doesn't match (its
leading Chinese province character is dropped; only the trailing 6 real
A-Z/0-9 characters are used, see dataset_ccpd.py). Treat this as a second
pretraining ingredient, not a real-data replacement for the actual
Indian fine-tuning set in train_lprnet.py.

Run from WSL2 (no CUDA-specific deps, but --device cuda is much faster):
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python pretrain_lprnet_synthetic.py --ccpd_root ../datasets/ccpd --ccpd_samples 40000

Then fine-tune on real data:
    python train_lprnet.py --init_ckpt checkpoints_lprnet_synth/lprnet_synth_final.pt
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from model_lprnet import LPRNet
from synthetic_plate_dataset import SyntheticPlateDataset
from dataset_ccpd import CCPDDataset
from dataset_lprnet import ctc_collate
from lprnet_vocab import NUM_CLASSES, BLANK, ctc_greedy_decode
from train_lprnet import levenshtein  # reuse the same CER implementation


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    n_exact, n_total = 0, 0
    total_cer_num, total_cer_den = 0, 0

    for imgs, targets, target_lengths, texts in loader:
        imgs = imgs.to(device)
        log_probs = torch.log_softmax(model(imgs), dim=1).permute(2, 0, 1)
        input_lengths = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item()
        n_batches += 1

        preds = ctc_greedy_decode(log_probs.cpu())
        for pred, gt in zip(preds, texts):
            n_total += 1
            if pred == gt:
                n_exact += 1
            total_cer_num += levenshtein(pred, gt)
            total_cer_den += max(len(gt), 1)

    model.train()
    return (total_loss / max(n_batches, 1), n_exact / max(n_total, 1),
            total_cer_num / max(total_cer_den, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15,
                     help="Each 'epoch' is one pass over --samples_per_epoch freshly "
                          "generated plates, not a fixed stored dataset.")
    ap.add_argument("--samples_per_epoch", type=int, default=30000)
    ap.add_argument("--val_samples", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt_dir", default="checkpoints_lprnet_synth")
    ap.add_argument("--ccpd_root", default=None,
                     help="Optional path to a directory of CCPD images (see "
                          "dataset_ccpd.py). When set, real CCPD crops are mixed "
                          "into every training epoch alongside the synthetic "
                          "plates -- real photographic variety, format caveats "
                          "and all (see this file's docstring).")
    ap.add_argument("--ccpd_samples", type=int, default=40000,
                     help="Cap on how many CCPD files to use (randomly subsampled "
                          "once at startup, fixed seed) -- keeps a 300k-image "
                          "download from dominating every epoch.")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Fixed seed for the val set only, so it's the same 2000 synthetic
    # plates every epoch (a stable yardstick) while train stays fresh
    # random plates every epoch (seed=None -> uses the shared `random`).
    synthetic_train_set = SyntheticPlateDataset(num_samples=args.samples_per_epoch, seed=None)
    val_set = SyntheticPlateDataset(num_samples=args.val_samples, seed=12345)

    if args.ccpd_root:
        ccpd_set = CCPDDataset(args.ccpd_root, max_samples=args.ccpd_samples, augment=True)
        train_set = ConcatDataset([synthetic_train_set, ccpd_set])
        print(f"train pairs: {len(synthetic_train_set)} synthetic + {len(ccpd_set)} CCPD "
              f"= {len(train_set)} | val pairs: {len(val_set)} (synthetic only)")
    else:
        train_set = synthetic_train_set
        print(f"train pairs: {len(train_set)} (synthetic only) | val pairs: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, collate_fn=ctc_collate, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, collate_fn=ctc_collate)

    model = LPRNet(num_classes=NUM_CLASSES).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch:3d}/{args.epochs}")
        for imgs, targets, target_lengths, _texts in pbar:
            imgs = imgs.to(args.device)

            opt.zero_grad(set_to_none=True)
            log_probs = torch.log_softmax(model(imgs), dim=1).permute(2, 0, 1)
            input_lengths = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            opt.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()

        train_loss = running / len(train_loader)
        val_loss, val_acc, val_cer = validate(model, val_loader, criterion, args.device)
        elapsed = time.time() - t0
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| val_acc {val_acc:.2%} | val_cer {val_cer:.4f} "
              f"| lr {sched.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"lprnet_synth_epoch{epoch}.pt"))

    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "lprnet_synth_final.pt"))
    print(f"done -> {args.ckpt_dir}/lprnet_synth_final.pt")
    print("Next: fine-tune on real data with:")
    print(f"  python train_lprnet.py --init_ckpt {args.ckpt_dir}/lprnet_synth_final.pt")


if __name__ == "__main__":
    main()
