"""
Physics-guided dehazing: use a model's output only to MEASURE the fog (thickness map t and veil colour A), then undo the fog on the real full-size
photo with the scattering law   I = J*t + A*(1-t)   ->   J = (I - A) / t + A.
The result is always a smooth, local re-scaling of the original photo: no tile seams, no invented texture, no colour cast the photo did not have.

  A  veil colour: mean colour of the haziest pixels (top 0.1% of the dark channel), clipped to a sane range
  t  per pixel by least squares across colour channels: t = sum((I-A)(Jm-A)) / sum((Jm-A)^2); where the model changed nothing (Jm ~ A) there is no
     information, so t is filled in from the neighbourhood (confidence-weighted blur), then smoothed edge-aware (guided filter, guide = the photo)
  J  (I - A) / max(t^gamma, t_floor) + A      gamma > 1 removes more fog (gamma < 1 less), t_floor caps the gain (noise amplification)

    python foggy_ft/clean/physics_dehaze.py --source ../datasets/foggy_photos_uvm_real --output ../datasets/foggy_photos_phys_real
"""
import argparse
import glob
import os
from multiprocessing import Pool

import cv2
import numpy as np

D = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets")) + os.sep


def box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided(guide, p, r, eps):
    """guided filter, single-channel guide (float32 0..1), single-channel p"""
    mi, mp = box(guide, r), box(p, r)
    cov = box(guide * p, r) - mi * mp
    var = box(guide * guide, r) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return box(a, r) * guide + box(b, r)


def estimate_A(small):
    dark = cv2.erode(small.min(2), np.ones((15, 15), np.uint8))
    k = max(10, int(0.001 * dark.size))
    idx = np.argpartition(dark.ravel(), -k)[-k:]
    A = small.reshape(-1, 3)[idx].mean(0)
    return np.clip(A, 0.55, 0.97)


def dehaze(hazy_bgr, model_bgr, gamma=1.0, t_floor=0.15, work=800, detail_cap=1.5, smooth=0.02, A_fixed=None):
    H, W = hazy_bgr.shape[:2]
    I = hazy_bgr.astype(np.float32) / 255.0
    Jm = cv2.resize(model_bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 if model_bgr.shape[:2] != (H, W) else model_bgr.astype(np.float32) / 255.0
    s = work / max(H, W)
    sw, sh = round(W * s), round(H * s)
    Il = cv2.resize(I, (sw, sh), interpolation=cv2.INTER_AREA)
    Jl = cv2.resize(Jm, (sw, sh), interpolation=cv2.INTER_AREA)
    A_est = estimate_A(Il)
    A = A_est if A_fixed is None else np.asarray(A_fixed, np.float32)       # video: a temporally smoothed veil colour avoids flicker
    n, d = Il - A, Jl - A
    w = (d * d).sum(2)
    t = (n * d).sum(2) / (w + 1e-6)
    t = np.clip(t, 0.05, 1.0)
    sig = (0.75 * smooth) * max(sw, sh)
    num = cv2.GaussianBlur(t * w, (0, 0), sig)
    den = cv2.GaussianBlur(w, (0, 0), sig)
    t_fill = num / (den + 1e-4)
    conf = np.clip(w / 0.01, 0, 1)                       # 1 where the model changed the pixel clearly, 0 where there is nothing to measure
    t = np.where(conf >= 1, t, conf * t + (1 - conf) * t_fill)
    gl = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    t = np.clip(guided(gl, t.astype(np.float32), max(8, int(smooth * max(sw, sh))), 1e-3), 0.05, 1.0)
    t_full = cv2.resize(t, (W, H), interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    t_full = np.clip(guided(g, t_full.astype(np.float32), 16, 1e-4), 0.05, 1.0)
    tt = np.maximum(t_full ** gamma, t_floor)[..., None]
    # base layer gets the full gain (restores broad contrast); fine detail + grain only up to detail_cap (stops noise blowing up in thick fog)
    base = np.stack([guided(g, I[..., c], 4, 4e-4) for c in range(3)], 2)
    J = (base - A) / tt + A + (I - base) * np.minimum(1.0 / tt, detail_cap)
    return np.clip(J * 255.0 + 0.5, 0, 255).astype(np.uint8), A_est


def _one(job):
    name, hazy_dir, src_dir, out_dir, gamma, t_floor, cap, smooth = job
    stem = os.path.splitext(name)[0]
    hz = cv2.imread(os.path.join(hazy_dir, name))
    md = cv2.imread(os.path.join(src_dir, stem + "_restored.png"))
    out, A = dehaze(hz, md, gamma, t_floor, detail_cap=cap, smooth=smooth)
    cv2.imwrite(os.path.join(out_dir, stem + "_restored.png"), out)
    return stem, A


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="folder of <stem>_restored.png model outputs used to measure the fog")
    ap.add_argument("--output", required=True)
    ap.add_argument("--hazy", default=D + "foggy_photos_upright")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--t_floor", type=float, default=0.15)
    ap.add_argument("--detail_cap", type=float, default=1.5)
    ap.add_argument("--smooth", type=float, default=0.02)
    ap.add_argument("--only", default="", help="comma-separated photo stems (quick test)")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(a.hazy, "*.jpg")) + glob.glob(os.path.join(a.hazy, "*.png")))
    if a.only:
        keep = set(a.only.split(","))
        names = [n for n in names if os.path.splitext(n)[0] in keep]
    jobs = [(n, a.hazy, a.source, a.output, a.gamma, a.t_floor, a.detail_cap, a.smooth) for n in names]
    with Pool(a.workers) as pool:
        for i, (stem, A) in enumerate(pool.imap_unordered(_one, jobs), 1):
            if i % 10 == 0 or i == len(jobs):
                print(i, "/", len(jobs), flush=True)
    print("done ->", a.output)
