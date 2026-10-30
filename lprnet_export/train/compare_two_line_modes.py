"""
Settles "which two-line layout is actually better?" without trusting a single
lucky/unlucky training run: trains every (layout, seed) combination on
IDENTICAL data and scores each on the same held-out test plates, then prints
the per-seed results and the mean per layout.

Why: one training run is noisy on this small dataset (swings of 3-4 plates on a
13-plate test are normal), so a single before/after comparison can't be trusted.

Data used: all scenes whose id starts with 'ds' (your own labeled photos:
ds6_*, ds9_*, ...) are forced into training, so validation is only the original
'crop_*' scenes -- the same yardstick for every run. The held-out test set is
datasets/heldout_test (built with label_review.py; it is never trained on).

Resumable: a (layout, seed) whose lprnet_avg.pt already exists is not retrained.
Roughly 8-10 minutes per run at the default 40 epochs.

Run from WSL2:
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python compare_two_line_modes.py --modes fixed,trim,hybrid --seeds 0,1,2
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys

GT = "../evaluation/baseline_ocr/ground_truth.json"


def run_train(mode, seed, args, force):
    out = os.path.join(args.out_root, f"{mode}_s{seed}")
    ckpt = os.path.join(out, "lprnet_avg.pt")
    log = os.path.join(out, "train.log")
    if os.path.exists(ckpt):
        print(f"[skip] {mode} seed {seed}: already trained")
        return out
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "train_lprnet.py",
           "--init_ckpt", args.init_ckpt, "--synthetic_mix", "1500",
           "--captcha_root", args.captcha_root, "--captcha_samples", "3000",
           "--force_train", force, "--seed", str(seed),
           "--epochs", str(args.epochs), "--avg_last", str(args.avg_last),
           "--ckpt_dir", out] + shlex.split(args.extra)
    print(f"[train] {mode} seed {seed} -> {out}", flush=True)
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                       env={**os.environ, "TWO_LINE_MODE": mode}, check=True)
    return out


def score(mode, out, args):
    r = subprocess.run([sys.executable, "label_review.py", "eval",
                        "--lprnet_ckpt", os.path.join(out, "lprnet_avg.pt"),
                        "--test_dir", args.test_dir],
                       capture_output=True, text=True, env={**os.environ, "TWO_LINE_MODE": mode})
    m = re.search(r"exact \(raw\) (\d+)/(\d+).*?CER ([0-9.]+)", r.stdout)
    val = None
    log_path = os.path.join(out, "train.log")
    if os.path.exists(log_path):
        v = re.search(r"averaged last .*?val_cer ([0-9.]+)", open(log_path).read())
        if v:
            val = float(v.group(1))
    if not m:
        print(r.stdout[-400:], r.stderr[-400:])
        return None
    return int(m.group(1)), int(m.group(2)), float(m.group(3)), val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="fixed,trim,hybrid")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--avg_last", type=int, default=10)
    ap.add_argument("--out_root", default="checkpoints_compare")
    ap.add_argument("--init_ckpt", default="checkpoints_lprnet_synth/lprnet_synth_final.pt")
    ap.add_argument("--captcha_root", default="/mnt/c/projects/jewwww/datasets/dataset5")
    ap.add_argument("--test_dir", default="../datasets/heldout_test")
    ap.add_argument("--extra", default="",
                    help="extra arguments passed through to train_lprnet.py, e.g. "
                         "\"--manifest ../datasets/dataset10_crops/manifest.csv --manifest_samples 1500\"")
    ap.add_argument("--min_test", type=int, default=10,
                    help="refuse to start if the held-out set has fewer plates than this")
    args = ap.parse_args()

    gt = json.load(open(GT))
    force = ",".join(k for k in gt if k.startswith("ds"))
    n_test = len(json.load(open(os.path.join(args.test_dir, "labels.json"))))
    print(f"forcing {len(force.split(','))} own-photo scenes into training; "
          f"scoring on {n_test} held-out plates\n", flush=True)
    if n_test < args.min_test:
        sys.exit(f"Only {n_test} held-out plates (need >= {args.min_test}): a test this small can't "
                 f"separate the layouts and the hours of training would be wasted. Add confirmed "
                 f"plates with `label_review.py ingest --test_frac 1.0` first, or lower --min_test.")

    modes = args.modes.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    results = {m: [] for m in modes}
    for mode in modes:
        for seed in seeds:
            out = run_train(mode, seed, args, force)
            res = score(mode, out, args)
            if res:
                results[mode].append((seed, res))
                print(f"   {mode} seed {seed}: held-out exact {res[0]}/{res[1]} | CER {res[2]:.3f} | val_cer {res[3]}", flush=True)

    print("\n=== summary (held-out test, exact match) ===")
    print(f"{'layout':<8} {'per-seed exact':<28} {'mean exact':>10} {'mean CER':>9} {'mean val_cer':>13}")
    for mode, rs in results.items():
        if not rs:
            continue
        exact = [r[1][0] for r in rs]
        tot = rs[0][1][1]
        cer = sum(r[1][2] for r in rs) / len(rs)
        vals = [r[1][3] for r in rs if r[1][3] is not None]
        print(f"{mode:<8} {str(exact):<28} {sum(exact) / len(exact):>6.1f}/{tot:<3} {cer:>9.3f} "
              f"{(sum(vals) / len(vals)) if vals else float('nan'):>13.4f}")
    print("\nOnly trust a layout if it wins on the mean across seeds by more than the seed-to-seed spread.")


if __name__ == "__main__":
    main()
