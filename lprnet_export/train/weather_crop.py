"""
Rain + fog + grain on ONE plate crop (PIL in, PIL out), for training the plate reader to read through weather.

The fog is a single thickness over the crop (a plate is at one distance), the drops are smaller than in a whole-scene photo, and
the grain has the measured softness and strength of the real foggy photos (see synth_rainfog.py). The crop is first brought to a
random width of 120-260 px: plates in the real foggy photos are about 230 px wide, and the grain/drop size must match that scale.
"""
import random

import cv2
import numpy as np
from PIL import Image

from synth_rainfog import degrade


def weather_degrade(img: Image.Image) -> Image.Image:
    w, h = img.size
    tw = random.randint(120, 260)
    if w != tw:
        img = img.resize((tw, max(8, int(round(h * tw / w)))), Image.BICUBIC)
    bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    out = degrade(bgr, np.random.default_rng(random.getrandbits(32)), crop_mode=True)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
