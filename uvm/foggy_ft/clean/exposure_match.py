"""Per-photo gamma so the mean luminance of the output reaches --target (the physics-strong output is clear but dark). Writes a separate folder."""
import argparse, glob, os
import cv2, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True); ap.add_argument("--output", required=True); ap.add_argument("--target", type=float, default=0.46)
a = ap.parse_args(); os.makedirs(a.output, exist_ok=True)
for f in sorted(glob.glob(os.path.join(a.input, "*_restored.png"))):
    im = cv2.imread(f); x = im.astype(np.float32) / 255
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    m = float(np.clip(g.mean(), 0.02, 0.98))
    gamma = np.log(a.target) / np.log(m)              # m**gamma = target
    gamma = float(np.clip(gamma, 0.6, 1.0))             # only brighten, never more than 0.6
    cv2.imwrite(os.path.join(a.output, os.path.basename(f)), np.clip(x ** gamma * 255 + 0.5, 0, 255).astype(np.uint8))
print("done ->", a.output)
