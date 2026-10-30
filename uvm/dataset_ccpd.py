"""
Real-photograph plate dataset built from CCPD (Chinese City Parking
Dataset, https://github.com/detectRecog/CCPD, MIT licensed, 300k+ images).

Why this exists: the project's own real data (dataset_lprnet.py) has only
~197 distinct plate texts -- nowhere near enough for a network to learn
general character SHAPES from real camera/lighting/lens conditions.
synthetic_plate_dataset.py solves the text-diversity problem but every
pixel is rendered, not photographed -- no real blur, glare, compression,
or lens distortion. CCPD is real dashcam/parking-lot photography at scale,
which fills that gap, even though its plates are Chinese-format, not
Indian-format.

Format mismatch and how it's handled: a CCPD plate is 7 characters --
1 leading Chinese province character, then 1 letter, then 5 letters/
digits. This project's alphabet (lprnet_vocab.py) is A-Z/0-9 only and has
no representation for the province character at all. So every sample
here DROPS the province character and uses only the trailing 6 (letter +
5 alphanumeric), which already fall inside A-Z/0-9 -- this dataset is a
real-photography character-shape source, not a full-plate-format match,
same spirit as how synthetic pretraining supplies text diversity without
being real. It should be mixed into PRETRAINING (alongside or instead of
synthetic data), not treated as equivalent to the real Indian fine-tuning
set in train_lprnet.py.

CCPD's own arrays/encoding (provinces/alphabets/ads index lists, filename
field layout) are exactly as published in the official repo's README.
The final array entry in `alphabets`/`ads` is CCPD's "unrecognized
character" placeholder (rendered as the letter 'O' by convention, which
is otherwise never used in the real alphabet/ads charsets to avoid 0/O
confusion) -- samples that hit it are skipped since it isn't a real
labeled character.

Filename format (dash-separated fields), e.g.:
    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    [0] area ratio
    [1] tilt degrees (horizontal_vertical)
    [2] bounding box: "left&top_right&bottom"
    [3] four vertex coords (unused here, bbox is enough for a crop)
    [4] plate number: 7 underscore-separated indices into
        provinces / alphabets / ads / ads / ads / ads / ads
    [5] brightness (unused)
    [6] blurriness (unused)
"""
import os
import random

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from lprnet_vocab import encode

INPUT_SIZE = (200, 24)  # (W, H) -- must match model_lprnet.py's expected input

# Exactly as published in the CCPD repo's README.
_PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
              "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
              "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
              "新", "警", "学", "O"]
_ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
              "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "O"]
_ADS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
        "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3",
        "4", "5", "6", "7", "8", "9", "O"]


def _parse_filename(fname):
    """Returns (bbox, text) or None if the filename doesn't match CCPD's
    format or its plate number includes the 'unrecognized character'
    placeholder. bbox is (left, top, right, bottom) ints; text is the
    trailing 6 real characters (province character dropped)."""
    stem = os.path.splitext(fname)[0]
    fields = stem.split("-")
    if len(fields) < 5:
        return None

    try:
        (left_top, right_bottom) = fields[2].split("_")
        left, top = (int(v) for v in left_top.split("&"))
        right, bottom = (int(v) for v in right_bottom.split("&"))

        idxs = [int(v) for v in fields[4].split("_")]
        if len(idxs) != 7:
            return None
        alphabet_char = _ALPHABETS[idxs[1]]
        ads_chars = [_ADS[i] for i in idxs[2:]]
    except (ValueError, IndexError):
        return None

    if alphabet_char == "O" or "O" in ads_chars:
        return None  # CCPD's own "unrecognized character" marker

    text = alphabet_char + "".join(ads_chars)
    return (left, top, right, bottom), text


class CCPDDataset(Dataset):
    def __init__(self, root, max_samples=None, seed=0, augment=False, input_size=INPUT_SIZE):
        """
        root: directory containing CCPD .jpg files -- searched recursively,
            since CCPD ships as several sub-folders (ccpd_base, ccpd_blur,
            ccpd_weather, ...). Only the filename is used for labels/bbox,
            so any mix of sub-folders under root works.
        max_samples: if set, randomly subsample the indexed list down to
            this many entries (fixed seed -- same subset every run) so a
            300k-image download doesn't dominate every pretraining epoch.
            None = use every valid file found.
        augment: mild brightness/contrast/rotation jitter, same spirit as
            LPRNetDataset._augment -- off by default since this is meant
            as a pretraining source where the raw photographic variety is
            already the point.
        """
        self.root = root
        self.augment = augment
        self.input_size = input_size
        self.entries = self._index(root)
        if not self.entries:
            raise RuntimeError(
                f"No usable CCPD-format filenames found under {root!r}. "
                f"Expected files like '025-95_113-154&383_386&473-...-0_0_22_27_27_33_16-37-15.jpg'."
            )
        if max_samples is not None and len(self.entries) > max_samples:
            rng = random.Random(seed)
            self.entries = rng.sample(self.entries, max_samples)
        self.to_tensor = transforms.ToTensor()

    def _index(self, root):
        entries = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                parsed = _parse_filename(fname)
                if parsed is None:
                    continue
                bbox, text = parsed
                entries.append((os.path.join(dirpath, fname), bbox, text))
        return entries

    def __len__(self):
        return len(self.entries)

    def _augment(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = transforms.functional.adjust_brightness(img, random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            img = transforms.functional.adjust_contrast(img, random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            img = img.rotate(random.uniform(-5, 5), fillcolor=(128, 128, 128), expand=False)
        return img

    def __getitem__(self, idx):
        path, bbox, text = self.entries[idx]
        img = Image.open(path).convert("RGB")
        img = img.crop(bbox)
        img = img.resize(self.input_size)

        if self.augment:
            img = self._augment(img)

        img_t = self.to_tensor(img)
        label = encode(text)
        return img_t, label, text


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "../datasets/ccpd"
    ds = CCPDDataset(root, max_samples=2000)
    print(f"Indexed {len(ds)} usable CCPD samples under {root!r} (capped at 2000 for this check).")
    out_dir = "ccpd_preview"
    os.makedirs(out_dir, exist_ok=True)
    for i in range(min(8, len(ds))):
        img_t, label, text = ds[i]
        img = transforms.ToPILImage()(img_t)
        img.save(os.path.join(out_dir, f"{i}_{text}.png"))
        print(f"{i}: text={text!r} label={label}")
    print(f"Saved previews to {out_dir}/ -- eyeball these to confirm the crop/label look right.")
