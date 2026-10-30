"""
dehaze_clahe.py
----------------
Local-contrast-based dehazing alternative to the DCP pipeline in
dehaze_baseline.py / dehaze_v2.py.

Why this exists: on dense/uniform fog (haze that isn't strongly
depth-varying), the Dark Channel Prior's core assumption - that some
patch in the true scene has a near-zero color channel - breaks down.
Recovering radiance from a violated prior amplifies noise instead of
structure, and on this project's heavy-fog test images it blew whole
frames to solid white (verified against dataset2/New Hazy dataset).

CLAHE sidesteps that: it never models atmospheric light or a
transmission map, so it can't diverge the way DCP recovery can. It
just redistributes local contrast in the L channel (LAB space) with a
clipped histogram, which held up on both light and heavy haze in
testing - including recovering plate text that the DCP baseline
destroyed and the raw hazy input couldn't quite show.
"""

import cv2


def apply_clahe(image, clip_limit: float = 3.0, tile_grid_size: int = 8):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
