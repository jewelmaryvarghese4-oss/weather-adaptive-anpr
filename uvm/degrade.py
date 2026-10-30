"""
Training-time "make this plate look like it came from a small, blurry video frame".

Why: LPRNet is trained on sharp, large plate crops, but a plate in a traffic
video is often only 30-60 px wide, soft, motion-smeared and JPEG/H.264
compressed. Sharpening or upscaling such a crop at inference time doesn't
bring the lost detail back (tested: sharpening made reads WORSE), so the
model has to learn to read degraded plates instead.

degrade_crop() takes a plate crop BEFORE it is resized to LPRNet's 200x24
input, shrinks it to a random small HEIGHT, then applies Gaussian blur,
optional motion blur and optional JPEG compression. The caller's normal resize
to 200x24 stretches it back up, exactly like a real small crop at inference.

Severity is capped on purpose: a sample blurred into a smear, paired with its
text label, would teach the CTC model to invent characters from nothing. The
shrink target is a HEIGHT (so wide single-line and flattened two-line crops are
treated alike), and the blur is scaled down as the crop gets smaller.
"""
import io
import random

import numpy as np
from PIL import Image, ImageFilter


def _motion_blur(img, length, angle):
    import cv2
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= max(k.sum(), 1e-6)
    return Image.fromarray(cv2.filter2D(np.asarray(img), -1, k, borderType=cv2.BORDER_REPLICATE))


def degrade_crop(img, min_h=12, max_h=30, min_w=16, blur_max=1.0, motion_p=0.25, jpeg_p=0.5):
    """img: RGB PIL crop. Returns a degraded copy (never larger than the input).
    A single-line video plate in our test clip is ~45x17 px, which sits inside this range."""
    img = img.convert("RGB")
    target_h = random.uniform(min_h, max_h)
    scale = min(1.0, target_h / img.height)
    new_w = max(min_w, int(round(img.width * scale)))
    new_h = max(8, int(round(img.height * scale)))
    if scale < 1.0 and new_w < img.width:
        img = img.resize((new_w, new_h), random.choice([Image.BILINEAR, Image.BICUBIC, Image.BOX]))
    small = min(1.0, img.height / 20.0)  # blur less on already-tiny crops
    r = random.uniform(0.0, blur_max) * small
    if r > 0.15:
        img = img.filter(ImageFilter.GaussianBlur(r))
    if random.random() < motion_p:
        img = _motion_blur(img, random.choice([3, 5]), random.uniform(0, 180))
    if random.random() < jpeg_p:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(30, 80))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img
