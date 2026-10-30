"""Pixel average of two output folders (<stem>_restored.png): blend_pair.py A B OUT [weight_of_A]"""
import glob, os, sys
import cv2, numpy as np
A, B, OUT = sys.argv[1:4]
w = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
os.makedirs(OUT, exist_ok=True)
fs = sorted(glob.glob(os.path.join(A, "*_restored.png")))
for f in fs:
    n = os.path.basename(f)
    a = cv2.imread(f).astype(np.float32); b = cv2.imread(os.path.join(B, n)).astype(np.float32)
    cv2.imwrite(os.path.join(OUT, n), np.clip(w * a + (1 - w) * b + 0.5, 0, 255).astype(np.uint8))
print(len(fs), "blended ->", OUT)
