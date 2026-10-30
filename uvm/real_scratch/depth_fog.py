"""
Option 2 (separate from every earlier model): depth-aware synthetic fog on REAL clear photos.
Fog follows the scattering law  I = J*t + A*(1-t)  with  t = exp(-beta * distance), where distance comes from a monocular depth map
(Depth Anything V2 Small, cached by depthfog_cache.py): far things fade first, near things stay readable -- like real mountain mist.
Also: fog patchiness, a veil colour A, forward-scattering blur (fog softens detail beyond the transmission loss), camera grain and JPEG.

    depth_fog(clear_bgr_u8, disp, rng, beta=(lo, hi)) -> foggy BGR uint8        disp: (H, W) float in [0,1], 1 = nearest
    beta = optical thickness of the farthest things (log-uniform in [lo, hi]); the nearest things get only `near` (default 22%) of it, so they always stay recoverable.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synth_rainfog import _smooth_noise, add_noise, add_raindrops


def disp_to_distance(disp):
    """relative distance in [0,1] from relative disparity: distance ~ 1/disparity, with a floor so the nearest pixels are not at 0"""
    d = 1.0 / (disp * 0.85 + 0.15)
    return (d - 1.0) / (1.0 / 0.15 - 1.0)


def depth_fog(clear_bgr_u8, disp, rng=None, beta=(0.6, 4.5), A_range=(0.58, 0.90), rain_prob=0.10, jpeg=True, near=0.22):
    rng = rng or np.random.default_rng()
    h, w = clear_bgr_u8.shape[:2]
    img = clear_bgr_u8.astype(np.float32) / 255.0
    dist = disp_to_distance(np.clip(disp.astype(np.float32), 0, 1))
    dist = dist ** rng.uniform(0.6, 1.0)                                       # how quickly the distance builds up
    patch = 0.75 + 0.5 * _smooth_noise(h, w, rng, 96)                           # fog is patchy
    ground = 1.0 + rng.uniform(0.0, 0.6) * np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None] * (1.0 if rng.random() < 0.5 else -1.0) * 0.5   # slightly denser low (or high)
    beta_v = float(np.exp(rng.uniform(np.log(beta[0]), np.log(beta[1]))))        # beta = optical thickness of the FARTHEST things (log-uniform): t_far = exp(-beta)
    tau = beta_v * (near + (1.0 - near) * dist) * patch * np.clip(ground, 0.6, 1.5)        # near things keep most of their content; the 0.08 floor: even they lose a little contrast
    t = np.exp(-tau)[..., None].astype(np.float32)
    g = rng.uniform(*A_range)
    A = np.array([g * rng.uniform(1.00, 1.08), g, g * rng.uniform(0.94, 1.00)], np.float32)       # BGR: neutral to slightly cool
    blur = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.8, 2.2))
    soft = (1.0 - t) * 0.6                                                        # thicker fog -> softer detail (forward scattering)
    base = img * (1.0 - soft) + blur * soft
    out = np.clip(base * t + A * (1.0 - t), 0.0, 1.0)
    if rng.random() < rain_prob:
        out = add_raindrops(out, rng)
    out = add_noise(out, rng)
    u8 = (out * 255.0 + 0.5).astype(np.uint8)
    if jpeg:
        ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 95))])
        if ok:
            u8 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return u8


def load_depth(path, size_hw):
    d = np.load(path).astype(np.float32)
    return cv2.resize(d, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)
