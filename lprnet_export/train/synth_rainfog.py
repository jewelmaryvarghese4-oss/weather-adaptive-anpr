"""
Synthetic "photo through a rainy window in fog": the kind of picture the dehazer fails on.

The real foggy phone photos have three things the training haze did not have:
  1. thick, uneven fog (bright veil, strongest far away),
  2. raindrops on the glass (small blurry lenses that bend the picture),
  3. camera noise that sits on top of the fog (so dehazing stretches it).
degrade() builds all three from a clear photo, so UVMNet can be fine-tuned on (degraded, clear) pairs and learn to remove the
fog WITHOUT stretching the noise or turning everything dark.

    python synth_rainfog.py --src some_folder_or_video --out preview.jpg     # look at samples
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np


def _smooth_noise(h, w, rng, scale):
    """Low-frequency random field in 0..1 (blotchy fog density)."""
    sh, sw = max(2, h // scale), max(2, w // scale)
    small = rng.random((sh, sw)).astype(np.float32)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.0), (w, h), interpolation=cv2.INTER_CUBIC)


def add_fog(img, rng, flat=False):
    """I = J*t + A*(1-t), with t = exp(-beta*depth). depth grows towards the top of the picture (far) and is blotchy."""
    h, w = img.shape[:2]
    ys = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]               # 1 at the top, 0 at the bottom
    if flat:       # a small crop (one plate) sits at one distance: nearly constant fog thickness over it
        depth = np.clip(rng.uniform(0.3, 1.1) * (0.92 + 0.16 * _smooth_noise(h, w, rng, 16)), 0.05, 1.6)
    else:
        depth = 0.35 + 0.65 * (ys ** rng.uniform(0.6, 1.4)) * np.ones((1, w), np.float32)
        depth = np.clip(depth * (0.6 + 0.8 * _smooth_noise(h, w, rng, 64)), 0.05, 1.6)
    beta = rng.uniform(0.8, 2.4)                                           # fog thickness (not so thick that nothing is left to recover)
    t = np.exp(-beta * depth)[..., None]
    # Measured on the 60 real photos (brightest 30% of the far third = the veil): mean RGB 0.72 / 0.75 / 0.78, brightness 0.67-0.83,
    # R/G 0.97, B/G 1.04 -- a neutral to slightly COOL grey, never warm. (The brown look of the real photos is the dirt ground seen
    # through the fog, not the fog itself.) Colours are BGR here.
    g = rng.uniform(0.66, 0.84)
    A = np.array([g * rng.uniform(1.00, 1.09), g, g * rng.uniform(0.93, 1.00)], np.float32)
    return np.clip(img * t + A * (1.0 - t), 0.0, 1.0), t


def add_raindrops(img, rng, small=False):
    """Drops on the glass: each is a small blurry lens that bends and softens what is behind it."""
    h, w = img.shape[:2]
    n = int(rng.uniform(0.4, 2.2) * h * w / 4000.0)
    m = np.zeros((h, w), np.float32)
    for _ in range(n):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(2, 7)) if small else int(rng.integers(3, 14))
        cv2.ellipse(m, (x, y), (r, int(r * rng.uniform(0.9, 1.8))), 0, 0, 360, 1.0, -1)
    # a few streaks running down
    for _ in range(int(n // 15)):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.line(m, (x, y), (x + int(rng.integers(-4, 5)), y + int(rng.integers(20, 90))), 1.0, int(rng.integers(1, 4)))
    m = cv2.GaussianBlur(m, (0, 0), 1.6)
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    k = rng.uniform(10.0, 28.0)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    bent = cv2.remap(img, xx + gx * k, yy + gy * k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    soft = cv2.GaussianBlur(bent, (0, 0), rng.uniform(2.0, 4.5))
    a = np.clip(m * 1.6, 0.0, 1.0)[..., None]
    out = bent * (1 - a) + soft * a
    return np.clip(out * (1.0 + 0.06 * a), 0.0, 1.0)                        # drops catch a little light


def add_noise(img, rng):
    """Camera noise AFTER the fog, so it is not thinned by the fog (this is what dehazing later stretches)."""
    # Measured on the real rainy photos (full size): the grain has a std of only about 1.7/255, but it is not fine-white --
    # neighbouring pixels are correlated (0.84 one pixel apart, 0.55 two, 0.23 three), like white noise blurred by sigma ~1.3.
    # Such soft grain looks like texture to the network, so it is NOT removed and then gets stretched by the dehazing.
    # (The first version used fine white noise 2-10x stronger than this, which the network learned to smooth away easily.)
    def grain(shape, std, blur):
        g = np.random.default_rng(int(rng.integers(1 << 30))).normal(0, 1.0, shape).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), blur)
        return g / (g.std() + 1e-6) * std

    h, w = img.shape[:2]
    std = rng.uniform(1.2, 3.5) / 255.0
    blur = rng.uniform(1.0, 1.8)
    lum = grain((h, w), std, blur)[..., None]
    chroma = np.stack([grain((h, w), std * rng.uniform(0.05, 0.3), blur) for _ in range(3)], -1)
    return np.clip(img + lum + chroma, 0.0, 1.0)


def degrade(clear_bgr_u8, rng=None, crop_mode=False):
    """clear BGR uint8 -> (foggy, rainy, noisy BGR uint8). Order matters: rain-on-glass, then fog seen through it is wrong
    physically (glass is nearest), so: fog first, then drops on the glass, then sensor noise, then JPEG."""
    rng = rng or np.random.default_rng()
    img = clear_bgr_u8.astype(np.float32) / 255.0
    # the real photos are underexposed compared with sunny video frames (mean brightness 0.63): darken the scene a little first
    img = np.clip(img * rng.uniform(0.7, 1.0), 0.0, 1.0) ** rng.uniform(1.0, 1.4)
    img, _ = add_fog(img, rng, flat=crop_mode)
    if rng.random() < 0.85:
        img = add_raindrops(img, rng, small=crop_mode)
    img = add_noise(img, rng)
    out = (img * 255.0 + 0.5).astype(np.uint8)
    q = int(rng.integers(70, 95))
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else out


def _frames(src, n, size_w=1500):
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.jp*g")) + glob.glob(os.path.join(src, "*.png")))
        for f in random.sample(files, min(n, len(files))):
            yield cv2.imread(f)
    else:
        cap = cv2.VideoCapture(src)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(total * 0.1, total * 0.9, n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                yield f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of clear photos, or a video file")
    ap.add_argument("--out", default="synth_preview.jpg")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    for f in _frames(a.src, a.n):
        s = 900 / f.shape[1]
        f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        rows.append(np.hstack([f, degrade(f, rng)]))
    cv2.imwrite(a.out, np.vstack(rows))
    print("wrote", a.out)
