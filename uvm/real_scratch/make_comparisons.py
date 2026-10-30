"""
finalout/fogrealcom/<n>.png : one comparison image per photo of finalin/foggyimages with three labelled panels side by side
  ORIGINAL (upright)  |  GoPro hybrid + tint fix (finalout/fogrealgopro)  |  Physics-guided dehaze, strong + exposure (finalout/fogrealphysics)
Panels are scaled to PANEL px on the long side.  Does not modify any of the source folders.
"""
import glob
import os

import cv2
import numpy as np

ROOT = "C:/projects/jewwww/"
ORIG = ROOT + "datasets/finalrun_foggyimages/upright/%s.png"
HYB = ROOT + "finalout/fogrealgopro/%s.png"
PHY = ROOT + "finalout/fogrealphysics/%s.png"
OUT = ROOT + "finalout/fogrealcom/"
PANEL = 1100
HEAD = 56
GAP = 10
LABELS = ["ORIGINAL", "GoPro hybrid + tint fix", "Physics-guided dehaze (strong + exposure)"]

os.makedirs(OUT, exist_ok=True)
names = sorted((os.path.basename(p)[:-4] for p in glob.glob(ORIG.replace("%s", "*"))), key=int)
for n in names:
    ims = [cv2.imread(P % n) for P in (ORIG, HYB, PHY)]
    if any(im is None for im in ims):
        print("skipped", n, [im is None for im in ims])
        continue
    s = PANEL / max(ims[0].shape[:2])
    ps = [cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA) for im in ims]
    h, w = ps[0].shape[:2]
    canvas = np.full((HEAD + h, 3 * w + 2 * GAP, 3), 30, np.uint8)
    for k, (p, lab) in enumerate(zip(ps, LABELS)):
        x = k * (w + GAP)
        canvas[HEAD:HEAD + h, x:x + w] = p
        cv2.putText(canvas, lab if k else "%s.png   ORIGINAL" % n, (x + 12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(OUT + n + ".png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])
print("comparison images:", len(os.listdir(OUT)))
