"""
Fog on the CLEAR Ponmudi photos, tuned so the result has the same haze / contrast / brightness as the REAL foggy Ponmudi photos.
(New file; synth_rainfog.py is not changed.) Uses synth_rainfog's raindrops and grain, but its own fog with an adjustable thickness.

    ponmudi_degrade(clear_bgr_u8, rng, beta=(lo, hi)) -> foggy rainy noisy BGR uint8
"""
import cv2
import numpy as np

from synth_rainfog import _smooth_noise, add_noise, add_raindrops


def add_fog_params(img, rng, beta, A_range=(0.66, 0.84), depth_gain=1.0):
    """I = J*t + A*(1-t), t = exp(-beta*depth). depth is larger towards the top of the picture (far) and blotchy, like the real photos."""
    h, w = img.shape[:2]
    ys = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
    depth = 0.35 + 0.65 * (ys ** rng.uniform(0.6, 1.4)) * np.ones((1, w), np.float32)
    depth = np.clip(depth * (0.6 + 0.8 * _smooth_noise(h, w, rng, 64)), 0.05, 1.6) * depth_gain
    t = np.exp(-rng.uniform(*beta) * depth)[..., None]
    g = rng.uniform(*A_range)
    A = np.array([g * rng.uniform(1.00, 1.09), g, g * rng.uniform(0.93, 1.00)], np.float32)     # BGR; neutral to slightly cool (measured)
    return np.clip(img * t + A * (1.0 - t), 0.0, 1.0)


def ponmudi_degrade(clear_bgr_u8, rng=None, beta=(1.5, 4.0), rain_prob=0.85, jpeg=True, **fog_kw):
    rng = rng or np.random.default_rng()
    img = clear_bgr_u8.astype(np.float32) / 255.0
    img = add_fog_params(img, rng, beta, **fog_kw)
    if rng.random() < rain_prob:
        img = add_raindrops(img, rng)
    img = add_noise(img, rng)
    out = (img * 255.0 + 0.5).astype(np.uint8)
    if jpeg:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 95))])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out
