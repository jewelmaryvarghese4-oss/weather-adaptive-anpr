"""
Loads C:\\projects\\jewwww\\datasets\\dataset5 -- the folder downloaded from
Kaggle's "Indian License Plate Images" listing (umar1103/final-licence).

IMPORTANT: despite the listing's title, this is NOT license plate
photography. Visual inspection of multiple samples (see conversation)
showed wavy, colored, distorted text with dot noise -- the output style
of a CAPTCHA generator (e.g. the `captcha` PyPI package), not plates:
no plate border, no plate background color, uniform 200x100 size across
all 30k+ files (real photos are never uniform size). Filenames are
still the ground-truth text (e.g. "KA 39 FY 2327.png"), so it's usable
as training data in principle, just a mismatched visual domain -- used
here on explicit request despite that mismatch, not because it's
expected to resemble deployment conditions the way CCPD or the real
fine-tuning set do.
"""
import os
import random

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from lprnet_vocab import ALPHABET, encode

INPUT_SIZE = (200, 24)  # (W, H) -- must match model_lprnet.py's expected input

_VALID_CHARS = set(ALPHABET)


class CaptchaStyleDataset(Dataset):
    def __init__(self, root, max_samples=None, seed=0, augment=False, input_size=INPUT_SIZE):
        """
        root: directory containing the image files (filename, minus
            extension and with spaces stripped, is the label).
        max_samples: if set, randomly subsample the indexed list down to
            this many entries (fixed seed) instead of using all ~30k.
        """
        self.input_size = input_size
        self.augment = augment
        self.entries = self._index(root)
        if not self.entries:
            raise RuntimeError(f"No usable labeled files found under {root!r}.")
        if max_samples is not None and len(self.entries) > max_samples:
            rng = random.Random(seed)
            self.entries = rng.sample(self.entries, max_samples)
        self.to_tensor = transforms.ToTensor()

    def _index(self, root):
        entries = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                text = os.path.splitext(fname)[0].upper().replace(" ", "")
                if not text or any(c not in _VALID_CHARS for c in text):
                    continue  # skip anything outside the A-Z/0-9 alphabet
                entries.append((os.path.join(dirpath, fname), text))
        return entries

    def __len__(self):
        return len(self.entries)

    def _augment(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = transforms.functional.adjust_brightness(img, random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            img = transforms.functional.adjust_contrast(img, random.uniform(0.7, 1.3))
        return img

    def __getitem__(self, idx):
        path, text = self.entries[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize(self.input_size)

        if self.augment:
            img = self._augment(img)

        img_t = self.to_tensor(img)
        label = encode(text)
        return img_t, label, text


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "../datasets/dataset5"
    ds = CaptchaStyleDataset(root, max_samples=2000)
    print(f"Indexed {len(ds)} usable samples under {root!r} (capped at 2000 for this check).")
    for i in range(5):
        _, label, text = ds[i]
        print(i, text, label)
