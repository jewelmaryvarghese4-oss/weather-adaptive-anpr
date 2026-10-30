"""
Trains LPRNet on this project's own labeled plate data (see
dataset_lprnet.py's docstring for where the ~1,930 image/text pairs come
from). Uses CTC loss -- the network never sees per-character bounding
boxes, just (image, full plate text) pairs; CTC handles the alignment.

--synthetic_mix (new): when set, every fine-tuning epoch's training pool
also includes this many freshly-generated synthetic plates alongside the
~1,741 real pairs, not just the real data alone. Why: a clean sequential
handoff (pretrain on synthetic, THEN fine-tune on only 197 real texts)
risks the network drifting away from the general character knowledge it
just learned and overfitting hard to those 197 texts specifically --
confirmed by the first pretrain+finetune run's train/val gap (train loss
0.05, val loss 1.5). Mixing fresh synthetic examples back in every epoch
keeps that general knowledge reinforced throughout fine-tuning instead of
letting it fade, while the real data still does the actual domain
adaptation (real camera/lighting/plate-format quirks synthetic can't
teach).

Run from WSL2 (works on CPU too -- LPRNet is tiny compared to UVMNet, no
mamba_ssm dependency, but --device cuda is much faster if available):
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python train_lprnet.py --init_ckpt checkpoints_lprnet_synth/lprnet_synth_final.pt --synthetic_mix 1500
"""
import argparse
import copy
import json
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from model_lprnet import LPRNet
from dataset_lprnet import LPRNetDataset, ctc_collate, INPUT_SIZE
from synthetic_plate_dataset import SyntheticPlateDataset
from lprnet_vocab import NUM_CLASSES, BLANK, ctc_greedy_decode


def levenshtein(a: str, b: str) -> int:
    """Edit distance -- used for CER (character error rate), a softer
    metric than exact-match accuracy: a 7-char plate read with 1 wrong
    character is "almost right", not simply wrong."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    n_exact, n_total = 0, 0
    total_cer_num, total_cer_den = 0, 0

    for imgs, targets, target_lengths, texts in loader:
        imgs = imgs.to(device)
        log_probs = torch.log_softmax(model(imgs), dim=1).permute(2, 0, 1)  # (T, B, C)
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
    avg_loss = total_loss / max(n_batches, 1)
    accuracy = n_exact / max(n_total, 1)
    cer = total_cer_num / max(total_cer_den, 1)
    return avg_loss, accuracy, cer


def seed_everything(seed):
    """Seeds python/numpy/torch RNGs so repeated runs on identical data
    start from the same point. Doesn't make runs bit-identical (CUDA's CTC
    loss is non-deterministic), but removes the largest source of
    run-to-run variation: unseeded shuffling, augmentation and the fresh
    synthetic plates generated each epoch."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def average_state_dicts(state_dicts):
    """Element-wise mean of float tensors across checkpoints; integer
    buffers (BatchNorm's num_batches_tracked) take the last checkpoint's."""
    avg = {}
    for k in state_dicts[0]:
        if state_dicts[0][k].is_floating_point():
            avg[k] = sum(sd[k] for sd in state_dicts) / len(state_dicts)
        else:
            avg[k] = state_dicts[-1][k]
    return avg


@torch.no_grad()
def recalibrate_bn(model, loader, device):
    """Averaged weights don't match any single checkpoint's BatchNorm
    running stats, so re-estimate them with one forward pass over the
    training data (cumulative average, momentum=None) -- the standard
    step when averaging weights (SWA)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for imgs, *_ in loader:
        model(imgs.to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../datasets/dataset")
    ap.add_argument("--ground_truth", default="../evaluation/baseline_ocr/ground_truth.json")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt_dir", default="checkpoints_lprnet")
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--init_ckpt", default=None,
                     help="Optional raw state_dict to start from -- e.g. "
                          "checkpoints_lprnet_synth/lprnet_synth_final.pt from "
                          "pretrain_lprnet_synthetic.py, so this fine-tunes an "
                          "already character-literate network instead of starting "
                          "from random weights and memorizing the small real dataset.")
    ap.add_argument("--synthetic_mix", type=int, default=0,
                     help="Add this many freshly-generated synthetic plates to the "
                          "training pool every epoch, alongside the real pairs -- "
                          "keeps general character knowledge reinforced during "
                          "fine-tuning instead of only ever seeing the 197 real "
                          "texts. 0 (default) = real data only, the original behavior.")
    ap.add_argument("--captcha_root", default=None,
                     help="Optional path to dataset_captcha_style.py's data (e.g. "
                          "../datasets/dataset5) -- mixes it into the training pool. "
                          "See dataset_captcha_style.py's docstring: despite its "
                          "source listing's name, this is CAPTCHA-style distorted "
                          "text, not plate photography -- a visual-domain mismatch, "
                          "included only because it was explicitly requested.")
    ap.add_argument("--captcha_samples", type=int, default=5000,
                     help="Cap on how many captcha_root files to use (random "
                          "subsample, fixed seed).")
    ap.add_argument("--force_train", default="",
                     help="Comma-separated scene_ids to force into the training "
                          "set regardless of the random val split -- e.g. a scene "
                          "whose fix keeps breaking because it happens to land in "
                          "val every time the deterministic shuffle is re-run with "
                          "unchanged data (the split only changes when the scene "
                          "count changes, not just by retraining).")
    ap.add_argument("--manifest", default=None,
                     help="Optional manifest.csv of extra (crop, text) pairs, e.g. from "
                          "prepare_zenodo.py -- mixed into the training pool.")
    ap.add_argument("--manifest_samples", type=int, default=0,
                     help="Cap on manifest rows used (random subsample, fixed seed). 0 = all.")
    ap.add_argument("--manifest_skip_groups", default="",
                     help="Comma-separated manifest 'group' values to leave out, e.g. "
                          "'car photo (C)' (those images can be the same photos as the original "
                          "dataset under a different label, which would leak validation scenes).")
    ap.add_argument("--manifest_verified_only", action="store_true",
                     help="Use only manifest rows whose label the model had confirmed (verified=1).")
    ap.add_argument("--seed", type=int, default=0,
                     help="Seed for all training randomness (shuffling, augmentation, "
                          "synthetic plates) so runs are comparable. Does NOT affect "
                          "the train/val split, which is fixed separately. -1 = unseeded.")
    ap.add_argument("--avg_last", type=int, default=0,
                     help="Average the weights of the last N epochs into one model, "
                          "saved as lprnet_avg.pt (BatchNorm stats re-estimated). "
                          "Smooths out the epoch-to-epoch flip-flopping on near-tie "
                          "characters. 0 = off. Try 15 with --epochs 60.")
    ap.add_argument("--weather_prob", type=float, default=0.0,
                     help="Chance (0-1) that a real/manifest training crop gets rain + fog + soft grain (weather_crop.py, "
                          "measured on the real foggy photos). Validation is never changed. 0 = off.")
    ap.add_argument("--degrade_prob", type=float, default=0.0,
                     help="Chance (0-1) that a REAL training crop (your labeled photos and the "
                          "--manifest pairs) is first shrunk to ~28-90 px wide, blurred and JPEG-"
                          "compressed (degrade.py), so the model learns to read small/blurry video "
                          "plates. Validation is never degraded. 0 = off (default). Try 0.5.")
    ap.add_argument("--heavy_repeats", action="store_true",
                     help="Synthetic plates get many more repeated digits (runs of 2-4, e.g. 7777) "
                          "and look-alike neighbours (B8, 0O, 5S ...). Targets the 'drops a repeated "
                          "or look-alike character' failure.")
    ap.add_argument("--synthetic_degrade_prob", type=float, default=0.0,
                     help="Chance (0-1) that a synthetic plate is also run through degrade.py "
                          "(small + blurry), so repeats are seen in the blurry form where they fail.")
    ap.add_argument("--synthetic_two_line", type=float, default=0.0,
                     help="Share (0-1) of synthetic plates rendered as TWO-LINE plates in Indian layouts "
                          "(KL 01 / AB 1234, KL 01 C / V 5776, KL 15 / 8194, KL 01 / CJ.145 ...) and flattened "
                          "like real ones. Targets dropped series letters at the line break of bike/auto plates.")
    ap.add_argument("--degrade_before_flatten", action="store_true",
                     help="With --degrade_prob: shrink two-line real crops BEFORE flattening (bike-plate size). "
                          "Default: after flattening, the recipe of the installed default model.")
    args = ap.parse_args()

    if args.seed >= 0:
        seed_everything(args.seed)
        print(f"seed: {args.seed}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    scene_ids = list(ground_truth.keys())
    rng = random.Random(0)
    rng.shuffle(scene_ids)
    n_val = max(1, int(len(scene_ids) * args.val_frac))
    val_ids = set(scene_ids[:n_val])
    train_ids = set(scene_ids[n_val:])

    forced = {s.strip() for s in args.force_train.split(",") if s.strip()}
    moved = forced & val_ids
    if moved:
        val_ids -= moved
        train_ids |= moved
        print(f"forced into train (were in val): {sorted(moved)}")
    missing = forced - moved - train_ids
    if missing:
        print(f"WARNING: --force_train scene_ids not found in ground_truth.json: {sorted(missing)}")

    print(f"scenes: {len(scene_ids)} total -> {len(train_ids)} train / {len(val_ids)} val")

    real_train_set = LPRNetDataset(args.data_root, ground_truth, scene_ids=train_ids, augment=True,
                                   degrade_prob=args.degrade_prob, weather_prob=args.weather_prob,
                                   degrade_before_flatten=args.degrade_before_flatten)
    val_set = LPRNetDataset(args.data_root, ground_truth, scene_ids=val_ids, augment=False)

    mix_sets = [real_train_set]
    mix_desc = [f"{len(real_train_set)} real"]

    if args.synthetic_mix > 0:
        # seed=None -> every __getitem__ call generates a fresh random
        # plate regardless of index or epoch, so reusing this one
        # instance across all epochs still means fresh synthetic
        # examples every time, not a fixed repeated subset.
        synthetic_mix_set = SyntheticPlateDataset(num_samples=args.synthetic_mix, seed=None,
                                                  heavy_repeats=args.heavy_repeats,
                                                  degrade_prob=args.synthetic_degrade_prob,
                                                  two_line_frac=args.synthetic_two_line)
        mix_sets.append(synthetic_mix_set)
        mix_desc.append(f"{args.synthetic_mix} synthetic/epoch"
                        + (f" ({args.synthetic_two_line:.0%} two-line)" if args.synthetic_two_line else ""))

    if args.captcha_root:
        from dataset_captcha_style import CaptchaStyleDataset
        captcha_set = CaptchaStyleDataset(args.captcha_root, max_samples=args.captcha_samples, augment=True)
        mix_sets.append(captcha_set)
        mix_desc.append(f"{len(captcha_set)} captcha-style")

    if args.manifest:
        from dataset_manifest import ManifestDataset
        manifest_set = ManifestDataset(args.manifest, max_samples=args.manifest_samples or None,
                                       verified_only=args.manifest_verified_only,
                                       skip_groups=tuple(g for g in args.manifest_skip_groups.split(",") if g),
                                       degrade_prob=args.degrade_prob, weather_prob=args.weather_prob,
                                       degrade_before_flatten=args.degrade_before_flatten)
        mix_sets.append(manifest_set)
        mix_desc.append(f"{len(manifest_set)} manifest")

    train_set = ConcatDataset(mix_sets) if len(mix_sets) > 1 else real_train_set
    print(f"train pairs: {' + '.join(mix_desc)} = {len(train_set)} | val pairs: {len(val_set)}")

    loader_gen = None
    if args.seed >= 0:
        loader_gen = torch.Generator()
        loader_gen.manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, collate_fn=ctc_collate, drop_last=True,
                               generator=loader_gen)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, collate_fn=ctc_collate)

    model = LPRNet(num_classes=NUM_CLASSES).to(args.device)
    if args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=args.device))
        print(f"initialized from {args.init_ckpt}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    best_cer = float("inf")
    avg_window = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch:3d}/{args.epochs}")
        for imgs, targets, target_lengths, _texts in pbar:
            imgs = imgs.to(args.device)

            opt.zero_grad(set_to_none=True)
            log_probs = torch.log_softmax(model(imgs), dim=1).permute(2, 0, 1)  # (T, B, C)
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

        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"lprnet_epoch{epoch}.pt"))
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "lprnet_best.pt"))
            print(f"  new best (val_cer={val_cer:.4f}) -> lprnet_best.pt")

        if args.avg_last > 0 and epoch > args.epochs - args.avg_last:
            avg_window.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    print(f"done -> {args.ckpt_dir}/lprnet_best.pt (best val_cer={best_cer:.4f})")

    if avg_window:
        avg_model = copy.deepcopy(model)
        avg_model.load_state_dict(average_state_dicts(avg_window))
        recalibrate_bn(avg_model, train_loader, args.device)
        a_loss, a_acc, a_cer = validate(avg_model, val_loader, criterion, args.device)
        avg_path = os.path.join(args.ckpt_dir, "lprnet_avg.pt")
        torch.save(avg_model.state_dict(), avg_path)
        print(f"averaged last {len(avg_window)} epochs -> {avg_path} "
              f"| val {a_loss:.4f} | val_acc {a_acc:.2%} | val_cer {a_cer:.4f}")
        print("use it with: --lprnet_ckpt " + avg_path)


if __name__ == "__main__":
    main()
