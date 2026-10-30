"""
Two-line plates (common on two-wheelers/autos in India -- e.g. "KL 01"
stacked above "BZ 6643" on one physical plate) get badly misread by
LPRNet because its input is fixed at 200x24, a wide single-line shape.
Force-resizing a roughly SQUARE two-line crop into that wide-short shape
vertically squashes both text rows together into noise.

Confirmed on dataset6: 4/4 checked two-line plates were badly misread
(e.g. real text 'KL01BZ6643' -> predicted '826043'), while every
correctly-read plate in the same set was single-line with a wide aspect
ratio. This isn't a model-knowledge gap (retraining alone won't fix a
preprocessing step that already destroyed the information) -- it's a
preprocessing mismatch, fixed here without touching the model at all.

is_two_line() flags a crop by aspect ratio: single-line plates in this
project run ~3.5-4:1 wide (e.g. 433x116, 1379x369); the two-line
failures found were ~1.0-1.3:1 (e.g. 328x265, 270x249, 236x184).

flatten_two_line() splits the crop top/bottom and lays the two halves
SIDE BY SIDE instead of stacked, turning the two-row plate into one wide
row -- which then resizes into LPRNet's 200x24 input the way a real
single-line plate does, reusing the existing trained model with no
architecture change.
"""
import os

import numpy as np
from PIL import Image

TWO_LINE_ASPECT_THRESHOLD = 2.5
MODE = os.environ.get("TWO_LINE_MODE", "fixed")  # 'fixed' | 'trim' | 'hybrid'
HYBRID_MIN_HEIGHT = 40


def is_two_line(img: Image.Image) -> bool:
    w, h = img.size
    return (w / h) < TWO_LINE_ASPECT_THRESHOLD


def flatten_two_line(img: Image.Image, gap_frac: float = 0.08) -> Image.Image:
    """Splits img into top/bottom halves and pastes them side by side,
    with a small gap between them -- without it, the abrupt edge where
    the two halves butt together sometimes gets misread as extra
    characters (confirmed: 1-3 spurious characters right at the seam
    across 4/4 test cases). The gap is filled with the plate's own
    background color (sampled from a corner) so it doesn't look like a
    hard edge needing to be read as something, just a blank separator."""
    w, h = img.size
    top = img.crop((0, 0, w, h // 2))
    bottom = img.crop((0, h // 2, w, h))
    gap = max(2, int(w * gap_frac))
    bg_color = img.getpixel((2, 2))
    flattened = Image.new("RGB", (w * 2 + gap, h // 2), bg_color)
    flattened.paste(top, (0, 0))
    flattened.paste(bottom, (w + gap, 0))
    return flattened


def _ink_mask(img: Image.Image):
    """Boolean mask of text-like pixels: strong deviation from the plate's
    median gray, so it works for dark-on-light AND light-on-dark plates."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    dev = np.abs(gray - np.median(gray))
    thr = max(25.0, 0.5 * float(np.percentile(dev, 97)))
    return dev > thr


def _trim_to_ink(img: Image.Image, mask, min_px=2):
    """Crops img (and its mask) to the bounding box of the ink, ignoring a
    thin side margin so the plate frame isn't mistaken for text. Falls
    back to the whole piece if no ink is found."""
    h, w = mask.shape
    m = max(1, int(0.03 * w))
    cols = np.where(mask[:, m:w - m].sum(axis=0) >= min_px)[0]
    if len(cols) == 0:
        return img
    x0, x1 = cols[0] + m, cols[-1] + m + 1
    rows = np.where(mask[:, x0:x1].sum(axis=1) >= min_px)[0]
    if len(rows) == 0:
        return img
    y0, y1 = rows[0], rows[-1] + 1
    pad = max(2, int(0.08 * (y1 - y0)))
    return img.crop((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)))


def flatten_two_line_trimmed(img: Image.Image, gap_frac: float = 0.5) -> Image.Image:
    """Smarter flattening: finds the real gap between the two text lines
    (lowest ink row in the middle band) instead of cutting at exactly 50%,
    trims each line to its own text so blank space/frame/screws don't eat
    the limited 200x24 input, scales both lines to the same height, and
    joins them left-to-right with a blank separator (gap_frac of the line
    height). A short top line like 'KL 27' no longer arrives as a small
    word floating in a mostly empty half."""
    img = img.convert("RGB")
    w, h = img.size
    mask = _ink_mask(img)

    # Row ink profile over the central columns, smoothed; split at its
    # valley within the middle band.
    c0, c1 = int(0.05 * w), int(0.95 * w)
    profile = mask[:, c0:c1].sum(axis=1).astype(np.float32)
    k = max(1, h // 20)
    smooth = np.convolve(profile, np.ones(k) / k, mode="same")
    lo, hi = int(0.30 * h), int(0.70 * h)
    split = lo + int(np.argmin(smooth[lo:hi])) if hi > lo else h // 2

    top = _trim_to_ink(img.crop((0, 0, w, split)), mask[:split])
    bottom = _trim_to_ink(img.crop((0, split, w, h)), mask[split:])

    target_h = max(8, int(round((top.height + bottom.height) / 2)))
    pieces = []
    for p in (top, bottom):
        pw = max(1, int(round(p.width * target_h / p.height)))
        pieces.append(p.resize((pw, target_h), Image.LANCZOS))

    gap = max(2, int(gap_frac * target_h))
    bg = tuple(int(v) for v in np.median(np.asarray(img).reshape(-1, 3), axis=0))
    out = Image.new("RGB", (pieces[0].width + gap + pieces[1].width, target_h), bg)
    out.paste(pieces[0], (0, 0))
    out.paste(pieces[1], (pieces[0].width + gap, 0))
    return out


def flatten_for_mode(img: Image.Image) -> Image.Image:
    """Flattens img as a two-line plate using the active MODE (always flattens,
    regardless of aspect ratio -- see normalize_plate_layout for the gated version)."""
    if MODE == "trim":
        return flatten_two_line_trimmed(img)
    if MODE == "hybrid":
        # Trimming needs enough pixels to find the gap between lines and crop
        # each line; on tiny crops (<~40 px tall, mostly squarish single-line
        # plates or badges that the aspect-ratio test also flags) it destroys
        # the text. Fall back to the plain fixed cut there.
        return flatten_two_line_trimmed(img) if img.height >= HYBRID_MIN_HEIGHT else flatten_two_line(img)
    return flatten_two_line(img)


def normalize_plate_layout(img: Image.Image) -> Image.Image:
    """Call this before LPRNet inference/training -- returns img unchanged for
    single-line plates, flattened for two-line ones. MODE 'fixed' is the
    original 50/50 split the current checkpoints were trained with;
    'trim' is flatten_two_line_trimmed (set TWO_LINE_MODE=trim, or assign
    two_line_plate.MODE). Train and infer with the SAME mode.

    The aspect-ratio test alone is ambiguous for crops around 1.9-2.6:1 (a short
    single-line plate and a wide two-line plate can both land there), so read_plate.py
    reads such crops BOTH ways and keeps the more plausible reading; this function
    is the plain gate used everywhere else (training data loading, etc.)."""
    if not is_two_line(img):
        return img
    return flatten_for_mode(img)
