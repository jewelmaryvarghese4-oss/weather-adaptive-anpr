"""
paired_dataset.py

Loads matched (hazy, clear) pairs from your datasets/dataset/{hazy,clear}
layout. Assumes: hazy/crop_0000_v3.jpg <-> clear/crop_0000.jpg — i.e. the
clear image's filename is the hazy filename with the trailing "_vN" variant
suffix stripped. If your actual naming differs, tell me and I'll adjust —
getting this pairing wrong would quietly reintroduce a version of the same
collapse problem (network "supervised" against the wrong clean image).
"""

import glob
import os
import random
import re

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

VARIANT_SUFFIX = re.compile(r"_v\d+$")


def hazy_to_clear_filename(hazy_filename: str) -> str:
    """crop_0000_v3.jpg -> crop_0000.jpg"""
    stem, ext = os.path.splitext(hazy_filename)
    clear_stem = VARIANT_SUFFIX.sub("", stem)
    return clear_stem + ext


class PairedHazeDataset(Dataset):
    def __init__(self, hazy_dir: str, clear_dir: str, resolution: int = 64,
                 limit: int | None = None, seed: int = 0):
        """
        limit: if set, randomly subsample down to this many (hazy, clear)
        pairs — e.g. for a quick fast-iteration run on 250 images instead
        of the full dataset, without needing a separate folder. Sampling
        is seeded for reproducibility (same 250 every run with the same
        seed), and happens AFTER the pairing/fail-fast check below, so a
        broken naming convention still errors loudly instead of just
        silently shrinking the dataset.
        """
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.resolution = resolution

        self.hazy_paths = sorted(
            glob.glob(os.path.join(hazy_dir, "*.jpg"))
            + glob.glob(os.path.join(hazy_dir, "*.png"))
        )
        if not self.hazy_paths:
            raise FileNotFoundError(f"No images found in {hazy_dir}")

        # fail fast if pairing is wrong, rather than silently training on
        # mismatched pairs
        missing = []
        for hp in self.hazy_paths:
            clear_name = hazy_to_clear_filename(os.path.basename(hp))
            if not os.path.exists(os.path.join(clear_dir, clear_name)):
                missing.append((os.path.basename(hp), clear_name))
        if missing:
            examples = missing[:5]
            raise FileNotFoundError(
                f"{len(missing)} hazy images have no matching clear image "
                f"using the 'strip _vN suffix' convention. Examples "
                f"(hazy -> expected clear): {examples}. Fix the naming "
                f"convention in hazy_to_clear_filename() if this doesn't "
                f"match your actual dataset layout."
            )

        if limit is not None and limit < len(self.hazy_paths):
            rng = random.Random(seed)
            self.hazy_paths = rng.sample(self.hazy_paths, limit)

    def __len__(self):
        return len(self.hazy_paths)

    def __getitem__(self, idx):
        hazy_path = self.hazy_paths[idx]
        clear_name = hazy_to_clear_filename(os.path.basename(hazy_path))
        clear_path = os.path.join(self.clear_dir, clear_name)

        hazy_img = Image.open(hazy_path).convert("RGB").resize(
            (self.resolution, self.resolution), Image.BILINEAR
        )
        clear_img = Image.open(clear_path).convert("RGB").resize(
            (self.resolution, self.resolution), Image.BILINEAR
        )

        return {
            "hazy": TF.to_tensor(hazy_img),
            "clear": TF.to_tensor(clear_img),
            "filename": os.path.basename(hazy_path),
        }


if __name__ == "__main__":
    # quick naming-convention sanity check without needing real data
    print(hazy_to_clear_filename("crop_0000_v3.jpg"), "should be crop_0000.jpg")
    print(hazy_to_clear_filename("crop_0123_v7.png"), "should be crop_0123.png")
