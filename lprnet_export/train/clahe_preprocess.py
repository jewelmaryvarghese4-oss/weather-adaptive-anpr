"""
CLAHE preprocessing for UVMNet's hazy input, reusing the exact recipe
already validated in clahe_dehazing/dehaze_clahe.py (LAB-space, clip_limit
3.0, 8x8 tile grid) -- that module's own docstring records it recovering
plate text that both the raw hazy input and the DCP baseline couldn't
show, and a fresh run of it on this project's hardest real-world test
photo (dusk, dense fog) produced a clean, grain-free result where every
UVMNet attempt so far has produced heavy grain artifacts.

The idea integrated here: apply CLAHE to the hazy input BEFORE UVMNet
sees it, at both train and inference time. The network currently has to
invent structure from a nearly flat, dark, low-contrast region when
correcting hard images -- that unconstrained guess is what's producing
grain. Giving it a locally-contrast-enhanced input first means whatever
real (if subtle) contrast already exists in the scene gets surfaced
before the network has to do anything, so its correction has something
concrete to build on instead of starting from near-nothing.

Only ever applied to the hazy/input side, never to the clear ground
truth target -- the target stays the untouched reference the model is
supervised against.
"""
import numpy as np
import cv2
from PIL import Image


def apply_clahe_pil(image: Image.Image, clip_limit: float = 3.0, tile_grid_size: int = 8) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)
