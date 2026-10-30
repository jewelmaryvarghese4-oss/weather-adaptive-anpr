"""
Fine-tunes UVMNet jointly on RESIDE-beta OTS (outdoor scenes), the Mendeley
Foggy-Hazy License Plates set (vehicle photos), and optionally NH-HAZE
(real non-homogeneous outdoor haze, 55 scenes) -- one run covers all
domains instead of separate fine-tunes overwriting each other's gains.
Same CombinedDehazeLoss recipe as train_ots.py / train_plates.py.

NH-HAZE is included specifically for general real-world fog robustness
(e.g. dense highway fog): neither OTS (synthetic haze) nor FoggyPlates
(real haze, but close-up vehicles) covers that scene type well, and it
showed up as a real weakness (color-cast/brightness issues) on off-domain
test photos. It's tiny (55 scenes) next to OTS/plates, so --nhhaze_repeat
duplicates its train split N times in the concatenated set so it's not
statistically invisible in every epoch.

Train/val split is done independently within each source dataset (by
scene/clear id, same reasoning as the single-dataset scripts: avoids
leaking near-duplicate variants across the split), then concatenated --
so val loss/PSNR reflects all domains, weighted by how many pairs each
contributes.

Also applies a synthetic low-light augmentation to a fraction of train
batches (see low_light_aug.py) -- none of the three real sources above
have any dusk/night examples, which showed up as grain artifacts on real
low-light fog test photos. --lowlight_prob 0 disables it.

Run from WSL2 (needs the real mamba_ssm CUDA kernel):
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python train_combined.py \
        --ots_root ../datasets/dataset4/OTS_BETA \
        --plates_root ../datasets/dataset4/FoggyPlates \
        --nhhaze_root ../datasets/dataset4/NH-HAZE
"""
import argparse
import os
import random
import time

import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from model_uvmnet import UVMNet
from dataset_ots import PairedOTSDataset
from dataset_foggyplates import PairedFoggyPlatesDataset
from dataset_nhhaze import PairedNHHazeDataset
from dataset_synth_rainfog import SynthRainFogDataset
from losses import CombinedDehazeLoss
from low_light_aug import augment_low_light


@torch.no_grad()
def psnr(pred, target):
    mse = torch.clamp((pred - target) ** 2, min=1e-10).mean()
    return -10.0 * torch.log10(mse)


@torch.no_grad()
def validate(model, loader, criterion, device, amp):
    # Deliberately NOT run under fp16 autocast, unlike training: validation
    # has no GradScaler to catch fp16 overflow, and the val loader isn't
    # shuffled -- one numerically fragile image (this model can emit large,
    # unstable corrections on off-domain data, see losses.py's
    # ResidualRegularizer docstring) would overflow to inf/nan on the exact
    # same image every single epoch and permanently poison the running
    # average from then on. fp32 removes that overflow risk; validation
    # isn't the throughput-critical path anyway.
    model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    n_skipped = 0
    for hazy, clear in loader:
        hazy, clear = hazy.to(device), clear.to(device)
        pred = model(hazy)
        loss, _parts = criterion(pred, clear, hazy)
        if not torch.isfinite(loss):
            n_skipped += hazy.size(0)
            continue
        pred_clamped = torch.clamp(pred.float(), 0.0, 1.0)
        total_loss += loss.item() * hazy.size(0)
        total_psnr += psnr(pred_clamped, clear).item() * hazy.size(0)
        n += hazy.size(0)
    model.train()
    if n_skipped:
        print(f"  WARNING: skipped {n_skipped} non-finite val sample(s) this epoch")
    if n == 0:
        return float("nan"), float("nan")
    return total_loss / n, total_psnr / n


def bn_buffer_snapshot(model):
    """Clone every BatchNorm running_mean/running_var buffer. Cheap --
    these are small per-channel vectors, not the conv weights."""
    return {k: v.detach().clone()
            for k, v in model.state_dict().items()
            if k.endswith("running_mean") or k.endswith("running_var")}


def bn_buffer_restore(model, snapshot):
    sd = model.state_dict()
    for k, v in snapshot.items():
        sd[k].copy_(v)


def split_ids(all_ids, val_frac, seed=0):
    rng = random.Random(seed)
    ids = list(all_ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    return set(ids[n_val:]), set(ids[:n_val])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ots_root", default="../datasets/dataset4/OTS_BETA",
                     help="Must contain <root>/clear/*.jpg and <root>/haze/*.jpg.")
    ap.add_argument("--plates_root", default="../datasets/dataset4/FoggyPlates",
                     help="Must contain <root>/clear/*.jpg and <root>/hazy/*.png.")
    ap.add_argument("--nhhaze_root", default=None,
                     help="Optional. Must contain <root>/<id>_GT.png and <root>/<id>_hazy.png "
                          "(flat, no subfolders). Omit to train without it.")
    ap.add_argument("--nhhaze_repeat", type=int, default=20,
                     help="Duplicate NH-HAZE's train split this many times in the "
                          "concatenated train set -- it's only ~52 train scenes vs. "
                          "OTS's thousands, so without oversampling it's statistically "
                          "invisible in every epoch.")
    ap.add_argument("--synth_clear_root", nargs="+", default=None,
                     help="Optional. Folders of CLEAR photos; fake rainy-window-in-fog versions of them are made on the "
                          "fly (synth_rainfog.py) and added as another training domain.")
    ap.add_argument("--synth_samples", type=int, default=3000, help="Fake-fog samples per epoch.")
    ap.add_argument("--synth_val", type=int, default=200, help="Fixed fake-fog validation samples.")
    ap.add_argument("--synth_hold", default="train6_",
                     help="Clear files whose path contains this are used ONLY for the fake-fog validation set.")
    ap.add_argument("--patch_size", type=int, default=256,
                     help="Shared crop size across both datasets -- must be the same "
                          "for both so batches can mix samples from either source.")
    ap.add_argument("--lowlight_prob", type=float, default=0.35,
                     help="Fraction of train batches (per-sample) darkened + noised to "
                          "simulate dusk/night conditions (see low_light_aug.py). None of "
                          "OTS/FoggyPlates/NH-HAZE have any low-light examples -- this is "
                          "what's missing to fix grain artifacts on real dusk/night fog "
                          "photos. 0 disables (matches previous runs' behavior).")
    ap.add_argument("--lowlight_dark_min", type=float, default=0.15)
    ap.add_argument("--lowlight_dark_max", type=float, default=0.55)
    ap.add_argument("--lowlight_noise_max", type=float, default=0.08)
    ap.add_argument("--clahe", action="store_true", default=False,
                     help="Apply CLAHE (LAB-space, clip 3.0, 8x8 tiles -- see "
                          "clahe_preprocess.py) to the hazy input before the model sees "
                          "it, at both train and inference time. Surfaces whatever local "
                          "contrast already exists in dark/low-contrast regions instead of "
                          "leaving the network to invent structure from near-nothing, which "
                          "is what's been producing grain on hard images. Applies to hazy "
                          "only, never to the clear target.")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--val_batch_size", type=int, default=8)
    ap.add_argument("--val_limit", type=int, default=300)
    ap.add_argument("--ots_limit", type=int, default=None,
                     help="Subsample OTS train pairs -- OTS (>15k pairs) otherwise "
                          "dwarfs the plates set (~1k pairs) in every epoch.")
    ap.add_argument("--ots_val_limit", type=int, default=200,
                     help="Cap OTS's own val split before concatenation. Without this, "
                          "OTS contributes ~3600 val pairs vs. plates' ~50 -- the global "
                          "--val_limit subsample then ends up ~99%% OTS, so val_psnr "
                          "barely reflects the plates/NH-HAZE domains at all.")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default="cuda", choices=["torch", "cuda"])
    ap.add_argument("--uvmb_size", type=int, default=32)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--init_ckpt", default="checkpoints_traffic/uvmnet_traffic_best_stateonly.pt")
    ap.add_argument("--ckpt_dir", default="checkpoints_combined")
    ap.add_argument("--ckpt_every", type=int, default=2)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--lambda_grad", type=float, default=0.15)
    ap.add_argument("--lambda_reg", type=float, default=0.02)
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    ots_train_ids, ots_val_ids = split_ids(PairedOTSDataset.all_clear_ids(args.ots_root), args.val_frac)
    plates_train_ids, plates_val_ids = split_ids(PairedFoggyPlatesDataset.all_ids(args.plates_root), args.val_frac)

    ots_train = PairedOTSDataset(args.ots_root, patch_size=args.patch_size, augment=True,
                                  clear_ids=ots_train_ids, limit=args.ots_limit, clahe=args.clahe)
    ots_val = PairedOTSDataset(args.ots_root, patch_size=args.patch_size, augment=False,
                                clear_ids=ots_val_ids, limit=args.ots_val_limit, clahe=args.clahe)
    plates_train = PairedFoggyPlatesDataset(args.plates_root, patch_size=args.patch_size, augment=True,
                                             ids=plates_train_ids, clahe=args.clahe)
    plates_val = PairedFoggyPlatesDataset(args.plates_root, patch_size=args.patch_size, augment=False,
                                           ids=plates_val_ids, clahe=args.clahe)

    print(f"OTS: {len(ots_train)} train / {len(ots_val)} val")
    print(f"FoggyPlates: {len(plates_train)} train / {len(plates_val)} val")
    if args.clahe:
        print("CLAHE preprocessing: enabled (applied to hazy input, train + val)")

    train_sets = [ots_train, plates_train]
    val_sets = [ots_val, plates_val]

    if args.nhhaze_root:
        nhhaze_train_ids, nhhaze_val_ids = split_ids(PairedNHHazeDataset.all_ids(args.nhhaze_root), args.val_frac)
        nhhaze_train = PairedNHHazeDataset(args.nhhaze_root, patch_size=args.patch_size, augment=True,
                                            ids=nhhaze_train_ids, clahe=args.clahe)
        nhhaze_val = PairedNHHazeDataset(args.nhhaze_root, patch_size=args.patch_size, augment=False,
                                          ids=nhhaze_val_ids, clahe=args.clahe)
        print(f"NH-HAZE: {len(nhhaze_train)} train / {len(nhhaze_val)} val "
              f"(train repeated x{args.nhhaze_repeat})")
        train_sets += [nhhaze_train] * args.nhhaze_repeat
        val_sets.append(nhhaze_val)

    if args.synth_clear_root:
        synth_train = SynthRainFogDataset(args.synth_clear_root, args.patch_size, args.synth_samples, augment=True,
                                          exclude=(args.synth_hold,))
        synth_val = SynthRainFogDataset(args.synth_clear_root, args.patch_size, args.synth_val, augment=False, seed=1,
                                        only=(args.synth_hold,))
        print(f"synthetic rain+fog: {len(synth_train.files)} clear train images x {args.synth_samples} samples/epoch, "
              f"{len(synth_val.files)} held-out images for {args.synth_val} val samples")
        train_sets.append(synth_train)
        val_sets.append(synth_val)

    train_set = ConcatDataset(train_sets)
    val_set = ConcatDataset(val_sets)
    if args.val_limit is not None and len(val_set) > args.val_limit:
        rng = random.Random(0)
        idx = rng.sample(range(len(val_set)), args.val_limit)
        val_set = torch.utils.data.Subset(val_set, idx)
    print(f"combined: {len(train_set)} train pairs | {len(val_set)} val pairs")
    if args.lowlight_prob > 0:
        print(f"low-light aug: {args.lowlight_prob:.0%} of train samples darkened "
              f"(darkness {args.lowlight_dark_min}-{args.lowlight_dark_max}, "
              f"noise up to {args.lowlight_noise_max}) -- val stays unaugmented")
    else:
        print("low-light aug: disabled")

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
        n_finite = 0
        n_skipped = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch:3d}/{args.epochs}")
        for hazy, clear in pbar:
            # Augment on CPU, before the .to(device) transfer -- even
            # without explicit sync points, WSL2's virtualized GPU has
            # real per-kernel-launch overhead, and this issues ~15 small
            # ops every batch. That was enough to turn ~0.45s/it into
            # ~4-5s/it. Doing it on CPU touches CUDA zero times; only the
            # one normal .to(device) transfer happens, same as before.
            hazy, clear = augment_low_light(
                hazy, clear, prob=args.lowlight_prob,
                darkness_range=(args.lowlight_dark_min, args.lowlight_dark_max),
                noise_std_range=(0.02, args.lowlight_noise_max),
            )
            hazy, clear = hazy.to(args.device), clear.to(args.device)

            # BatchNorm updates its running_mean/running_var as a side
            # effect of THIS forward call, via an exponential moving
            # average, regardless of whether the loss below turns out
            # finite -- and once a nan enters an EMA it never clears
            # (nan * 0.9 + anything * 0.1 is still nan, forever). A run
            # hit exactly this: one bad batch poisoned 12 BN buffers in
            # the decoder, silently breaking every eval()-mode forward
            # pass (i.e. all of validate()) for the rest of training
            # while train loss looked fine (train-mode BatchNorm uses
            # each batch's own live stats, not the corrupted running
            # average, so the corruption was invisible there). Snapshot
            # before, restore after if this batch turns out bad, so a
            # skipped batch can't leave permanent damage behind.
            bn_snapshot = bn_buffer_snapshot(model)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu",
                                 dtype=torch.float16, enabled=scaler.is_enabled()):
                pred = model(hazy)
                loss, parts = criterion(pred, clear, hazy)

            if not torch.isfinite(loss):
                bn_buffer_restore(model, bn_snapshot)
                n_skipped += 1
                pbar.set_postfix(loss="nan(skip)", pix="-")
                continue

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            for k in running_parts:
                running_parts[k] += parts[k].item()
            n_finite += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", pix=f"{parts['charbonnier'].item():.4f}")
        sched.step()

        if n_skipped:
            print(f"  WARNING: skipped {n_skipped} non-finite train batch(es) this epoch")
        train_loss = running / max(n_finite, 1)
        val_loss, val_psnr = validate(model, val_loader, criterion, args.device, scaler.is_enabled())
        if val_loss != val_loss:  # nan != nan -- every val sample was non-finite
            raise RuntimeError(
                f"epoch {epoch}: ALL val samples were non-finite -- something is still "
                f"broken (check BN buffers with the diagnostic snippet). Stopping now "
                f"instead of silently burning hours on a run whose validation is dead. "
                f"Last known-good checkpoint should be the most recent one before this."
            )
        elapsed = time.time() - t0
        n_batches = max(n_finite, 1)
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
            path = os.path.join(args.ckpt_dir, f"uvmnet_combined_epoch{epoch}.pt")
            torch.save(state, path)
            print(f"  saved -> {path}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state["best_val_loss"] = best_val_loss
            torch.save(state, os.path.join(args.ckpt_dir, "uvmnet_combined_best.pt"))

    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "uvmnet_combined_final.pt"))
    print(f"done -> {os.path.join(args.ckpt_dir, 'uvmnet_combined_final.pt')}")


if __name__ == "__main__":
    main()
