"""
Trains a FRESH UVMNet (random weights, nothing inherited from any earlier model) on REAL fog images only:
  GoProHazy real fog / clear road frames (paired by place, not pixel aligned)  +  your real Ponmudi fog photos and video frames.
No synthetic fog, no rain/grain simulation, no OTS/RESIDE. Loss: real_losses.py (statistics that survive misalignment; no pixel-wise target).
The last layer is zero-initialised so the untrained network is the identity (output = input); everything else starts random.

    python real_scratch/train_uvm_real.py --root ../datasets/real_scratch --out_dir checkpoints_real_scratch --epochs 20
SEPARATE depth-fog variant of train_uvm_real.py (that script is unchanged). Resumes by itself if <out_dir>/real_resume.pt exists (run the same command again). Every epoch prints no-reference numbers on held-out GoPro
test videos and saves preview_epochN.jpg (4 of your hardest Ponmudi photos: foggy | output) so you can watch it learn; stop any time with Ctrl+C.
"""
import argparse
import csv
import copy
import glob
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from model_uvmnet import UVMNet
from dataset_real import RealDataset
from real_losses import real_loss
from losses import CombinedDehazeLoss

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="../datasets/real_scratch")
ap.add_argument("--out_dir", default="checkpoints_real_scratch")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--samples", type=int, default=8000, help="random crops per epoch")
ap.add_argument("--batch", type=int, default=6)
ap.add_argument("--crop", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--gopro_frac", type=float, default=0.6, help="share of paired GoPro crops (the rest = your Ponmudi fog)")
ap.add_argument("--ema", type=float, default=0.999)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--uvmb_size", type=int, default=32)
ap.add_argument("--save_every", type=int, default=5)
ap.add_argument("--preview_dir", default="../datasets/newfoggyphotos/results/upright")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--init_ckpt", default=None, help="OPTIONAL: start from this checkpoint instead of random weights (NOT used for the fresh real-only model)")
ap.add_argument("--gpu_fraction", type=float, default=0.78, help="cap on the GPU memory this process may reserve: a cache that grows to the whole 8 GB makes Windows swap GPU memory and the run crawls")
ap.add_argument("--nhhaze_root", default=None, help="NH-HAZE folder (55 real haze-machine pairs, pixel aligned): turns on the aligned pixel-loss pass")
ap.add_argument("--aligned_synth_samples", type=int, default=0, help="also use synthetic Ponmudi fog pairs (clear Ponmudi photos + calibrated fog) in the aligned pass: samples per epoch")
ap.add_argument("--aligned_batch", type=int, default=2)
ap.add_argument("--aligned_lambda", type=float, default=3.0, help="weight of the aligned pixel loss")
ap.add_argument("--aligned_nh_frac", type=float, default=0.5, help="share of NH-HAZE crops in the aligned pass (rest = synthetic Ponmudi fog)")
ap.add_argument("--aligned_gopro_synth_frac", type=float, default=0.0, help="share of the aligned pass that is heavy synthetic fog on real clear GoPro frames (0 = off)")
ap.add_argument("--synth_beta_hi", type=float, default=6.0)
ap.add_argument("--real_loss_scale", type=float, default=1.0, help="multiplies the whole real-photo (statistics) loss; <1 lets the aligned ground-truth pixel loss dominate")
ap.add_argument("--depthfog_root", default=None, help="datasets/depthfog (depthfog_cache.py output): turns on depth-aware fog on real clear photos in the aligned pass")
ap.add_argument("--aligned_depth_frac", type=float, default=0.8, help="share of the aligned pass that is depth-aware fog on real clear photos")
ap.add_argument("--depth_beta_lo", type=float, default=0.6)
ap.add_argument("--depth_beta_hi", type=float, default=4.5)
ap.add_argument("--depth_pon_frac", type=float, default=0.4, help="share of depth-fog samples made on the clear Ponmudi photos (the rest: GoPro road frames)")
ap.add_argument("--max_steps", type=int, default=0, help="stop after N steps (smoke test)")
a = ap.parse_args()


class SynthOnClearFrames(torch.utils.data.Dataset):
    """aligned pairs made by putting a random HEAVY synthetic fog on real CLEAR road frames. The fog differs every time: thickness (beta), how strongly it
    grows with distance (depth_gain), airlight brightness and colour, patchiness. The true clear image is the answer.
    fixed=True -> the same crops and the same fog every time (a held-out test with a known ground truth)."""
    def __init__(self, paths, crop=256, beta=(0.8, 6.0), n=10000, fixed=False, seed=0):
        self.paths, self.crop, self.beta, self.n, self.fixed, self.seed = paths, crop, beta, n, fixed, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import cv2 as _cv2
        from synth_ponmudi import ponmudi_degrade
        rnd = random.Random(self.seed * 1_000_003 + i) if self.fixed else random
        rng = np.random.default_rng(self.seed * 1_000_003 + i) if self.fixed else np.random.default_rng()
        im = _cv2.imread(self.paths[rnd.randrange(len(self.paths))])
        s = rnd.uniform(0.55, 1.0)
        im = _cv2.resize(im, None, fx=s, fy=s, interpolation=_cv2.INTER_AREA)
        h, w = im.shape[:2]
        y0, x0 = rnd.randint(0, h - self.crop), rnd.randint(0, w - self.crop)
        clear = im[y0:y0 + self.crop, x0:x0 + self.crop]
        if rnd.random() < 0.5:
            clear = clear[:, ::-1].copy()
        lo = self.beta[0]
        bad = ponmudi_degrade(clear, rng, beta=(rnd.uniform(lo, self.beta[1] * 0.6), rnd.uniform(self.beta[1] * 0.6, self.beta[1])), rain_prob=0.3,
                              A_range=(rnd.uniform(0.55, 0.75), rnd.uniform(0.78, 0.92)), depth_gain=rnd.uniform(0.7, 2.2))
        t = lambda a: torch.from_numpy(_cv2.cvtColor(np.ascontiguousarray(a), _cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        return t(bad), t(clear)


class DepthFogOnClear(torch.utils.data.Dataset):
    """aligned pairs for the SEPARATE depth-fog model: depth-aware fog (depth_fog.py, depth from Depth Anything V2 Small) on REAL clear photos
    (clear Ponmudi photos and GoPro road frames). The true clear crop is the answer. fixed=True -> same crops and same fog every time (held-out test)."""
    def __init__(self, rows, crop=256, beta=(0.6, 4.5), n=10000, fixed=False, seed=0, pon_frac=0.4):
        self.pon = [r for r in rows if r[2] == "ponmudi"]
        self.gop = [r for r in rows if r[2] != "ponmudi"]
        self.crop, self.beta, self.n, self.fixed, self.seed, self.pon_frac = crop, beta, n, fixed, seed, pon_frac

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import cv2 as _cv2
        from depth_fog import depth_fog, load_depth
        rnd = random.Random(self.seed * 1_000_003 + i) if self.fixed else random
        rng = np.random.default_rng(self.seed * 1_000_003 + i) if self.fixed else np.random.default_rng()
        pool = self.pon if (self.pon and (not self.gop or rnd.random() < self.pon_frac)) else self.gop
        ip, dp, grp = pool[rnd.randrange(len(pool))]
        im = _cv2.imread(ip)
        if grp == "ponmudi" and max(im.shape[:2]) > 1600:
            k = 1600 / max(im.shape[:2])
            im = _cv2.resize(im, None, fx=k, fy=k, interpolation=_cv2.INTER_AREA)
        s = rnd.uniform(0.6, 1.0) if grp == "ponmudi" else rnd.uniform(0.55, 1.0)
        im = _cv2.resize(im, None, fx=s, fy=s, interpolation=_cv2.INTER_AREA)
        h, w = im.shape[:2]
        if h < self.crop or w < self.crop:
            k = self.crop / min(h, w) * 1.01
            im = _cv2.resize(im, None, fx=k, fy=k, interpolation=_cv2.INTER_CUBIC)
            h, w = im.shape[:2]
        d = load_depth(dp, (h, w))
        y0, x0 = rnd.randint(0, h - self.crop), rnd.randint(0, w - self.crop)
        clear = im[y0:y0 + self.crop, x0:x0 + self.crop]
        dc = d[y0:y0 + self.crop, x0:x0 + self.crop]
        if rnd.random() < 0.5:
            clear, dc = clear[:, ::-1].copy(), dc[:, ::-1].copy()
        bad = depth_fog(np.ascontiguousarray(clear), np.ascontiguousarray(dc), rng, beta=self.beta)
        t = lambda a_: torch.from_numpy(_cv2.cvtColor(np.ascontiguousarray(a_), _cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        return t(bad), t(clear)


class AlignedMix(torch.utils.data.Dataset):
    """aligned (hazy, clear) crops: NH-HAZE (real haze machine) or synthetic Ponmudi fog"""
    def __init__(self, nh, syn, nh_frac, n, gp=None, gp_frac=0.0):
        self.nh, self.syn, self.nh_frac, self.n, self.gp, self.gp_frac = nh, syn, nh_frac, n, gp, gp_frac

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.gp is not None and random.random() < self.gp_frac:
            return self.gp[0][0]
        if self.nh is not None and (self.syn is None or random.random() < self.nh_frac):
            return self.nh[random.randrange(len(self.nh))]
        return self.syn[random.randrange(len(self.syn))]


def main():
    global dev, model, ema, ds, loader, opt, sched, scaler, root, val, prev, it

    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    root = os.path.abspath(a.root)

    model = UVMNet(n_channels=3, backend="cuda", uvmb_size=a.uvmb_size).to(dev)
    torch.nn.init.zeros_(model.outc.conv.weight)
    torch.nn.init.zeros_(model.outc.conv.bias)
    if a.init_ckpt:
        model.load_state_dict(torch.load(a.init_ckpt, map_location=dev))
        print("initialised from", a.init_ckpt, flush=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    print("fresh UVMNet: %.1f M parameters, random init (output layer zero -> starts as identity)" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    ds = RealDataset(root, crop=a.crop, samples=a.samples, gopro_frac=a.gopro_frac, seed=a.seed)
    print("real training data: %d GoPro matched pairs | %d real Ponmudi foggy images | %d real clear references (%d Ponmudi + %d GoPro frames)" % (
        len(ds.pairs), len(ds.hazy), len(ds.pclear) + len(ds.gclear), len(ds.pclear), len(ds.gclear)), flush=True)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    aligned_iter, aligned_val, crit = None, None, None
    if a.nhhaze_root or a.aligned_synth_samples:
        import random as _r
        from torch.utils.data import Dataset
        from dataset_nhhaze import PairedNHHazeDataset
        DD = os.path.join(os.path.dirname(HERE), "..", "datasets")
        nh_all = sorted(PairedNHHazeDataset.all_ids(a.nhhaze_root)) if a.nhhaze_root else []
        _r.Random(0).shuffle(nh_all)
        nh_val_ids, nh_train_ids = set(nh_all[:5]), set(nh_all[5:])
        nh_tr = PairedNHHazeDataset(a.nhhaze_root, patch_size=a.crop, augment=True, ids=nh_train_ids) if nh_all else None
        nh_va = PairedNHHazeDataset(a.nhhaze_root, patch_size=a.crop, augment=False, ids=nh_val_ids) if nh_all else None
        syn_tr = syn_va = None
        if a.aligned_synth_samples:
            from dataset_ponmudi_ms import PonmudiMultiScaleDataset
            args_ = (os.path.join(DD, "gdrive_extra_photos"), os.path.join(DD, "ponmudi_clear", "list.csv"), os.path.join(DD, "ponmudi_clear", "boxes.csv"), a.crop)
            syn_tr = PonmudiMultiScaleDataset(*args_, a.aligned_synth_samples, augment=True)
            syn_va = PonmudiMultiScaleDataset(*args_, 40, augment=False, seed=1)

        gp_ds, gp_frac_used, depth_va = None, a.aligned_gopro_synth_frac, []
        if a.depthfog_root:
            import csv as _csv
            man = [(r["image_path"], r["depth_path"], r["group"]) for r in _csv.DictReader(open(os.path.join(a.depthfog_root, "manifest.csv"), encoding="utf-8"))]
            pon_all = sorted(r for r in man if r[2] == "ponmudi")
            pon_val, pon_tr = pon_all[-4:], pon_all[:-4]                         # 4 clear Ponmudi photos are never trained on
            tr_rows = [r for r in man if r[2] == "gopro_train"] + pon_tr
            gp_ds = [DepthFogOnClear(tr_rows, a.crop, (a.depth_beta_lo, a.depth_beta_hi), a.samples, pon_frac=a.depth_pon_frac)]
            gp_frac_used = a.aligned_depth_frac
            depth_va = [DepthFogOnClear([r for r in man if r[2] == "gopro_val"], a.crop, (a.depth_beta_lo, a.depth_beta_hi), 60, fixed=True, seed=11),
                        DepthFogOnClear(pon_val, a.crop, (a.depth_beta_lo, a.depth_beta_hi), 24, fixed=True, seed=13)]
            print("depth-aware fog: %d clear GoPro frames + %d clear Ponmudi photos (+%d held-out Ponmudi photos), %.0f%% of the aligned pass, thickness tau_far %.1f-%.1f" % (
                sum(1 for r in tr_rows if r[2] != "ponmudi"), len(pon_tr), len(pon_val), 100 * gp_frac_used, a.depth_beta_lo, a.depth_beta_hi), flush=True)
        elif a.aligned_gopro_synth_frac > 0:
            clear_paths = sorted({r["clear_path"] for r in __import__("csv").DictReader(open(os.path.join(root, "gopro_train_pairs.csv"), encoding="utf-8"))})
            gp_ds = [SynthOnClearFrames(clear_paths, a.crop, (0.8, a.synth_beta_hi), a.samples)]
            print("heavy synthetic fog on %d real clear GoPro frames: %.0f%% of the aligned pass, thickness beta 0.8-%.1f" % (len(clear_paths), 100 * a.aligned_gopro_synth_frac, a.synth_beta_hi), flush=True)
        aloader = DataLoader(AlignedMix(nh_tr, syn_tr, a.aligned_nh_frac, a.samples, gp_ds, gp_frac_used), batch_size=a.aligned_batch, shuffle=False, num_workers=2, drop_last=True, persistent_workers=True)

        def _forever():
            while True:
                for b in aloader:
                    yield b
        aligned_iter = _forever()
        crit = CombinedDehazeLoss(lambda_grad=0.15, lambda_reg=0.02).to(dev)
        vclear = sorted({r["clear_path"] for r in __import__("csv").DictReader(open(os.path.join(root, "gopro_val_pairs.csv"), encoding="utf-8"))})
        heavy_va = SynthOnClearFrames(vclear, a.crop, (2.0, 6.0), 60, fixed=True, seed=7)
        val_sets = (nh_va, syn_va) + tuple(depth_va) if depth_va else (nh_va, syn_va, heavy_va)
        aligned_val = [DataLoader(d, batch_size=4, shuffle=False, num_workers=0) for d in val_sets if d is not None]
        print("aligned pass ON: NH-HAZE %s train / %s held-out scenes | synthetic Ponmudi fog %s | batch %d, weight %.1f" % (
            len(nh_tr) if nh_tr else 0, len(nh_va) if nh_va else 0, "on" if syn_tr else "off", a.aligned_batch, a.aligned_lambda), flush=True)
    steps_per_epoch = a.samples // a.batch
    total = steps_per_epoch * a.epochs
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(total, 1))))))
    scaler = torch.amp.GradScaler("cuda")
    start_epoch, step = 1, 0
    rp = os.path.join(a.out_dir, "real_resume.pt")
    if os.path.exists(rp):
        ck = torch.load(rp, map_location=dev, weights_only=False)      # our own checkpoint (the scheduler state holds numpy scalars)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        print("resumed from epoch", ck["epoch"], flush=True)
        del ck                       # the loaded checkpoint (1.9 GB of copies) stayed on the GPU for the whole run and pushed the peak over the limit
        torch.cuda.empty_cache()


    def bn_snap(m):
        return {k: v.clone() for k, v in m.state_dict().items() if "running_" in k or "num_batches" in k}


    def bn_restore(m, snap):
        sd = m.state_dict()
        for k, v in snap.items():
            sd[k].copy_(v)


    @torch.no_grad()
    def ema_update():
        d = a.ema
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        for be, bm in zip(ema.buffers(), model.buffers()):
            be.copy_(bm)


    # ---- validation + preview helpers
    def to_t(bgr):
        return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().div(255).to(dev)


    def to_bgr(t):
        return cv2.cvtColor((t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


    @torch.no_grad()
    def run(net, bgr):
        h, w = bgr.shape[:2]
        x = to_t(bgr)
        ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate")
        with torch.autocast("cuda", dtype=torch.float16):
            y = net(x)
        return to_bgr(y[:, :, :h, :w].float())


    def metrics(bgr):
        f = bgr.astype(np.float32) / 255
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        return [float(cv2.erode(f.min(2), np.ones((15, 15), np.uint8)).mean()), float(g.std()), float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255), float(g.mean())]


    def small(bgr, w):
        s = w / max(bgr.shape[:2])
        return cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr


    val_rows = list(csv.DictReader(open(os.path.join(root, "gopro_val_pairs.csv"), encoding="utf-8")))[:40]
    val = [(small(cv2.imread(r["hazy_path"]), 768), small(cv2.imread(r["clear_path"]), 768)) for r in val_rows]
    prev = sorted(glob.glob(os.path.join(a.preview_dir, "*.jpg")))[:4] or [r[0] for r in [(x["path"],) for x in csv.DictReader(open(os.path.join(root, "ponmudi_hazy.csv")))]][:4]
    prev = [small(cv2.imread(p), 1024) for p in prev]


    def validate(epoch):
        ema.eval()
        crit_vals = []
        if aligned_val:
            for name, dl in zip(("NH-HAZE held-out scenes", "synthetic Ponmudi fog (fixed)", "DEPTH fog, held-out roads", "DEPTH fog, held-out Ponmudi") if depth_va else ("NH-HAZE held-out scenes", "synthetic Ponmudi fog (fixed)", "HEAVY fog on held-out roads"), aligned_val):
                ps, bs = [], []
                with torch.no_grad():
                    for vh, vc in dl:
                        vh, vc = vh.to(dev), vc.to(dev)
                        with torch.autocast("cuda", dtype=torch.float16):
                            vp = ema(vh)
                        mse = ((vp.float().clamp(0, 1) - vc) ** 2).mean((1, 2, 3)).clamp(min=1e-10)
                        ps += (-10 * torch.log10(mse)).tolist()
                        bs += (-10 * torch.log10(((vh - vc) ** 2).mean((1, 2, 3)).clamp(min=1e-10))).tolist()
                print("  aligned PSNR, %-30s: output %.2f dB  (the hazy input itself: %.2f dB)" % (name, float(np.mean(ps)), float(np.mean(bs))), flush=True)
                if name.startswith("HEAVY") or name.startswith("DEPTH"):
                    crit_vals.append(float(np.mean(ps)))
        mh, mo, mc = [], [], []
        for h, c in val:
            o = run(ema, h)
            mh.append(metrics(h)); mo.append(metrics(o)); mc.append(metrics(c))
        mh, mo, mc = (np.mean(m, 0) for m in (mh, mo, mc))
        print("  held-out GoPro test videos (%d frames), no-reference numbers:   haze   contrast  saturation  brightness" % len(val), flush=True)
        for n, m in (("hazy input", mh), ("this model's output", mo), ("clear frames (the target look)", mc)):
            print("    %-32s %6.3f %9.3f %10.3f %11.3f" % (n, *m), flush=True)
        rows = [np.hstack([p, cv2.resize(run(ema, p), (p.shape[1], p.shape[0]))]) for p in prev]
        w = min(r.shape[1] for r in rows)
        cv2.imwrite(os.path.join(a.out_dir, "preview_epoch%d.jpg" % epoch), np.vstack([r[:, :w] for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        torch.cuda.empty_cache()                   # give the validation blocks back: a cache full of odd-sized blocks made the next epoch crawl (GPU memory full)
        return float(np.mean(crit_vals)) if crit_vals else None


    best_heavy = [-1e9]
    if start_epoch == 1:
        validate(0)
        print("  (epoch 0 = untrained identity network: output equals input)", flush=True)
    it = iter(loader)

    def bn_bad(m):
        return any(v.dtype.is_floating_point and not bool(torch.isfinite(v).all()) for k, v in m.state_dict().items() if "running_" in k)

    def repair_bn(n_batches=120):
        """recompute the BatchNorm running statistics from the training data (the weights are untouched)"""
        bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        for m in bns:
            m.reset_running_stats()
            m.momentum = None                      # cumulative average over the batches below
        model.train()
        wbad = [k for k, v in model.state_dict().items() if "running_" not in k and "num_batches" not in k and v.dtype.is_floating_point and not bool(torch.isfinite(v).all())]
        print("model weights with NaN/Inf:", len(wbad), wbad[:3], flush=True)
        skipped = 0
        with torch.no_grad():
            for _ in tqdm(range(n_batches), desc="repairing BatchNorm statistics", unit="batch", dynamic_ncols=True):
                srcs = [next(it)[0]] + ([next(aligned_iter)[0]] if aligned_iter is not None else [])
                for hz in srcs:
                    snap_r = bn_snap(model)
                    with torch.autocast("cuda", dtype=torch.float16):
                        out_ = model(hz.to(dev))
                    if not bool(torch.isfinite(out_).all()) or any(not bool(torch.isfinite(v).all()) for v in snap_r.values() if v.dtype.is_floating_point) and False:
                        bn_restore(model, snap_r)
                        skipped += 1
                    elif bn_bad(model):
                        bn_restore(model, snap_r)
                        skipped += 1
        print("skipped %d batches whose activations were not finite" % skipped, flush=True)
        for m in bns:
            m.momentum = 0.1
        for be, bm in zip(ema.buffers(), model.buffers()):
            be.copy_(bm)
        print("BatchNorm statistics repaired; finite now:", not bn_bad(model) and not bn_bad(ema), flush=True)

    if bn_bad(model) or bn_bad(ema):
        print("NaN found in BatchNorm statistics -> repairing", flush=True)
        repair_bn()
    for epoch in range(start_epoch, a.epochs + 1):
        model.train()
        t0 = time.time()
        sums, n = {}, 0
        pbar = tqdm(range(steps_per_epoch), desc="epoch %d/%d" % (epoch, a.epochs), unit="step", dynamic_ncols=True, smoothing=0.05)
        for i in pbar:
            try:
                hazy, ref, paired = next(it)
            except StopIteration:
                it = iter(loader)
                hazy, ref, paired = next(it)
            hazy, ref, paired = hazy.to(dev, non_blocking=True), ref.to(dev, non_blocking=True), paired.to(dev)
            snap = bn_snap(model)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(hazy)
            loss, parts = real_loss(pred, ref, hazy, paired)
            loss = loss * a.real_loss_scale
            if not torch.isfinite(loss):
                bn_restore(model, snap)
                tqdm.write("  non-finite loss, step skipped")
                continue
            scaler.scale(loss).backward()
            if aligned_iter is not None:
                ah, ac = next(aligned_iter)
                ah, ac = ah.to(dev, non_blocking=True), ac.to(dev, non_blocking=True)
                snap_a = bn_snap(model)
                with torch.autocast("cuda", dtype=torch.float16):
                    ap_ = model(ah)
                al, _ = crit(ap_.float(), ac, ah)
                if torch.isfinite(al):
                    scaler.scale(a.aligned_lambda * al).backward()
                    sums["aligned"] = sums.get("aligned", 0) + al.item()
                else:
                    bn_restore(model, snap_a)          # keep NaN out of the BatchNorm running statistics
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema_update()
            step += 1
            n += 1
            sums["loss"] = sums.get("loss", 0) + loss.item()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0) + v
            pbar.set_postfix(loss="%.3f" % (sums["loss"] / n), lr="%.1e" % sched.get_last_lr()[0], gpu="%.1fG" % (torch.cuda.memory_allocated() / 2**30))
            if a.max_steps and step >= a.max_steps:
                break
        pbar.close()
        print("epoch %2d | loss %.4f (%s) | lr %.2e | %.0fs" % (epoch, sums.get("loss", 0) / max(n, 1), " ".join("%s %.3f" % (k, v / max(n, 1)) for k, v in sums.items() if k != "loss"), sched.get_last_lr()[0], time.time() - t0), flush=True)
        if bn_bad(model) or bn_bad(ema):
            print("NaN found in BatchNorm statistics at the end of epoch %d -> repairing" % epoch, flush=True)
            repair_bn()
        hp = validate(epoch)
        if hp is not None and hp > best_heavy[0]:          # keep the best epoch by the held-out heavy-fog ground-truth PSNR (it swings 1-3 dB from epoch to epoch)
            best_heavy[0] = hp
            torch.save(ema.state_dict(), os.path.join(a.out_dir, "best_val_ema.pt"))
            print("  new best held-out fog PSNR (mean of the held-out fog sets) %.2f dB (epoch %d) -> best_val_ema.pt" % (hp, epoch), flush=True)
        torch.save(ema.state_dict(), os.path.join(a.out_dir, "real_latest_ema.pt"))
        if epoch % a.save_every == 0 or epoch == a.epochs:
            torch.save(ema.state_dict(), os.path.join(a.out_dir, "real_epoch%d_ema.pt" % epoch))
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch, "step": step}, rp)
        if a.max_steps and step >= a.max_steps:
            break
    torch.save(ema.state_dict(), os.path.join(a.out_dir, "real_final_ema.pt"))
    print("done -> %s/real_final_ema.pt" % a.out_dir, flush=True)


if __name__ == "__main__":
    main()
