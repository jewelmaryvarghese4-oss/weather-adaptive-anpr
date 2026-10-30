"""
Generic (crop image, text) dataset read from a manifest.csv with columns
path,text[,verified,...] -- paths are relative to the manifest's folder.
Written for prepare_zenodo.py's output, but works for any CSV like that.

Applies the same preprocessing as the real-data loader (two-line flattening,
resize to LPRNet's 200x24, mild brightness/contrast/rotation jitter) so mixed
batches look consistent.
"""
import csv
import os
import random
import re

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from degrade import degrade_crop
from lprnet_vocab import encode
from two_line_plate import is_two_line, normalize_plate_layout

INPUT_SIZE = (200, 24)
_VALID = re.compile(r"^[A-Z0-9]{3,12}$")


class ManifestDataset(Dataset):
    def __init__(self, manifest_csv, max_samples=None, seed=0, verified_only=False,
                 skip_groups=(), augment=True, input_size=INPUT_SIZE, degrade_prob=0.0,
                 degrade_before_flatten=False, weather_prob=0.0):
        self.input_size = input_size
        self.augment = augment
        self.degrade_prob = degrade_prob  # chance of degrade.degrade_crop() per training sample
        self.degrade_before_flatten = degrade_before_flatten
        self.weather_prob = weather_prob  # chance of rain+fog+grain (weather_crop.py) per training sample
        base = os.path.dirname(os.path.abspath(manifest_csv))
        entries = []
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                text = re.sub(r"[^A-Z0-9]", "", (r.get("text") or "").upper())
                if not _VALID.match(text):
                    continue
                if verified_only and str(r.get("verified", "1")) != "1":
                    continue
                if (r.get("group") or "") in skip_groups:
                    continue
                entries.append((os.path.join(base, r["path"]), text))
        if not entries:
            raise RuntimeError(f"No usable rows in {manifest_csv!r}")
        if max_samples and len(entries) > max_samples:
            entries = random.Random(seed).sample(entries, max_samples)
        self.entries = entries
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.entries)

    def _augment(self, img):
        if random.random() < 0.5:
            img = transforms.functional.adjust_brightness(img, random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            img = transforms.functional.adjust_contrast(img, random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            img = img.rotate(random.uniform(-5, 5), fillcolor=(128, 128, 128), expand=False)
        return img

    def __getitem__(self, idx):
        path, text = self.entries[idx]
        img = Image.open(path).convert("RGB")
        if self.augment and self.weather_prob > 0 and random.random() < self.weather_prob:
            from weather_crop import weather_degrade
            img = weather_degrade(img)
        degrade = self.augment and self.degrade_prob > 0 and random.random() < self.degrade_prob
        if degrade and self.degrade_before_flatten:
            two = is_two_line(img)   # shrink BEFORE flattening, see dataset_lprnet.py
            img = degrade_crop(img, min_h=18 if two else 12, max_h=40 if two else 30)
        img = normalize_plate_layout(img)
        if degrade and not self.degrade_before_flatten:
            img = degrade_crop(img, min_w=32)  # recipe of the installed default model
        img = img.resize(self.input_size)
        if self.augment:
            img = self._augment(img)
        return self.to_tensor(img), encode(text), text
