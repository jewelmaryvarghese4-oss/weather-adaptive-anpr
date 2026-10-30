"""
Pulls back the warm (tan/brown) cast the dehazer adds in dense fog.
The correction lives only in the LOW-frequency chroma (Lab a*/b*, Gaussian sigma ~3% of the long side), so the colours of objects (tail lights, signs,
plates) are kept; it is weighted by how hazy the INPUT is there (dark channel of the hazy photo), so clear parts of a photo are left alone.
Luminance is never touched.   beta = 0 -> no change, 1 -> low-frequency chroma returns fully to the input's in the densest fog.
"""
import cv2
import numpy as np


def _smooth(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def tint_field(hazy_bgr, out_bgr, beta=0.7, lo=0.45, hi=0.80, work=768):
    """returns the (da, db) Lab correction at working resolution, shape (h, w, 2) for the working size, plus that size."""
    h, w = out_bgr.shape[:2]
    s = min(1.0, work / max(h, w))
    sz = (max(1, round(w * s)), max(1, round(h * s)))
    hs = cv2.resize(hazy_bgr, sz, interpolation=cv2.INTER_AREA)
    os_ = cv2.resize(out_bgr, sz, interpolation=cv2.INTER_AREA)
    dark = cv2.erode(hs.astype(np.float32).min(2) / 255.0, np.ones((15, 15), np.uint8))
    sig = 0.03 * max(sz)
    wgt = cv2.GaussianBlur(_smooth(dark, lo, hi), (0, 0), sig)
    ab_in = cv2.cvtColor(hs.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)[..., 1:]
    ab_out = cv2.cvtColor(os_.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)[..., 1:]
    cast = cv2.GaussianBlur(ab_out, (0, 0), sig) - cv2.GaussianBlur(ab_in, (0, 0), sig)
    return -beta * wgt[..., None] * cast, sz


def fix_tint(hazy_bgr, out_bgr, beta=0.7, **kw):
    delta, sz = tint_field(hazy_bgr, out_bgr, beta, **kw)
    h, w = out_bgr.shape[:2]
    delta = cv2.resize(delta, (w, h), interpolation=cv2.INTER_CUBIC) if (w, h) != sz else delta
    lab = cv2.cvtColor(out_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    lab[..., 1:] += delta
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
