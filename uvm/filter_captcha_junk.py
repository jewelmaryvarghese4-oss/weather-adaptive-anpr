"""
One-off cleanup: moves images from datasets/dataset5/ALL that look like
CAPTCHA-generator output (wavy colored text, dot noise -- see this
conversation's finding that dataset5 is a MIX of CAPTCHA junk, plain
text templates, plate-styled renders, and genuine real photos, not
uniformly one or the other) into datasets/dataset5/_captcha_quarantine,
instead of deleting them -- reversible if the heuristic needs tuning.

Heuristic (calibrated against confirmed examples from this conversation,
~83% precision on a spot-check): CAPTCHA images have a near-white
(brightness > 0.88), near-flat but NOT perfectly solid background
(corner std-dev between 4 and 32) -- the generator's background noise
dots create slight unevenness that a true solid-color plate render
(std ~0) or a genuinely textured real photo (std > 40) doesn't have.

Run from WSL2:
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python filter_captcha_junk.py
"""
import colorsys
import os
import shutil

from PIL import Image

ROOT = "../datasets/dataset5/ALL"
QUARANTINE = "../datasets/dataset5/_captcha_quarantine"


def corner_stats(path):
    img = Image.open(path).convert("RGB").resize((100, 50))
    w, h = img.size
    patch = 10
    corners = []
    for (x0, y0) in [(0, 0), (w - patch, 0), (0, h - patch), (w - patch, h - patch)]:
        for x in range(x0, x0 + patch):
            for y in range(y0, y0 + patch):
                corners.append(img.getpixel((x, y)))
    rs = [p[0] for p in corners]
    gs = [p[1] for p in corners]
    bs = [p[2] for p in corners]
    mean = (sum(rs) / len(rs), sum(gs) / len(gs), sum(bs) / len(bs))
    var = sum((r - mean[0]) ** 2 + (g - mean[1]) ** 2 + (b - mean[2]) ** 2 for r, g, b in corners) / len(corners)
    _, s, v = colorsys.rgb_to_hsv(mean[0] / 255, mean[1] / 255, mean[2] / 255)
    return var ** 0.5, s, v


def is_captcha_like(path):
    try:
        std, s, v = corner_stats(path)
    except Exception:
        return False
    return 4 <= std <= 32 and s < 0.12 and v > 0.88


def main():
    os.makedirs(QUARANTINE, exist_ok=True)
    files = [f for f in os.listdir(ROOT) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    moved = 0
    for f in files:
        path = os.path.join(ROOT, f)
        if is_captcha_like(path):
            shutil.move(path, os.path.join(QUARANTINE, f))
            moved += 1

    print(f"moved {moved}/{len(files)} suspected CAPTCHA-style images to {QUARANTINE}")
    print(f"remaining in {ROOT}: {len(files) - moved}")


if __name__ == "__main__":
    main()
