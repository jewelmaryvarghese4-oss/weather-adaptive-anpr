"""
Paired clear/hazy dataset for datasets/dataset (ANPR plate/vehicle crops).

Layout:
    <root>/clear/crop_XXXX.jpg
    <root>/hazy/crop_XXXX_vN.jpg   (N = 0..7, 8 synthetic haze variants per scene)

Crops are tiny and wildly non-square (median ~85x29px, some as small as
8x5px) -- far smaller than any sensible training crop, so unlike
dataset_traffic.py this resizes (doesn't crop) both images to a fixed
square size, same approach the original train_uvmnet.py used for this
same data.
"""
import os
import re
import random

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from haze_aug import augment_haze_density


class PairedPlateDataset(Dataset):
    def __init__(self, root, resize=128, augment=True, scene_ids=None,
                 density_aug=False, density_alpha_range=(0.0, 1.3)):
        self.clear_dir = os.path.join(root, "clear")
        self.hazy_dir = os.path.join(root, "hazy")
        self.resize = resize
        self.augment = augment
        self.density_aug = density_aug
        self.density_alpha_range = density_alpha_range
        self.pairs = self._index_pairs(scene_ids)
        if not self.pairs:
            raise RuntimeError(f"No hazy/clear pairs found under {root!r}.")
        self.tf_resize = transforms.Resize((resize, resize))
        self.to_tensor = transforms.ToTensor()

    def _index_pairs(self, scene_ids):
        pairs = []
        for hf in sorted(os.listdir(self.hazy_dir)):
            stem, ext = os.path.splitext(hf)
            m = re.match(r"^(crop_\d+)_v\d+$", stem)
            if not m:
                continue
            scene = m.group(1)
            if scene_ids is not None and scene not in scene_ids:
                continue
            clear_path = os.path.join(self.clear_dir, scene + ext)
            if os.path.exists(clear_path):
                pairs.append((clear_path, os.path.join(self.hazy_dir, hf)))
        return pairs

    @staticmethod
    def all_scene_ids(root):
        clear_dir = os.path.join(root, "clear")
        ids = []
        for f in sorted(os.listdir(clear_dir)):
            stem, _ext = os.path.splitext(f)
            ids.append(stem)
        return ids

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clear_path, hazy_path = self.pairs[idx]
        clear = self.tf_resize(Image.open(clear_path).convert("RGB"))
        hazy = self.tf_resize(Image.open(hazy_path).convert("RGB"))

        if self.augment and random.random() < 0.5:
            clear = clear.transpose(Image.FLIP_LEFT_RIGHT)
            hazy = hazy.transpose(Image.FLIP_LEFT_RIGHT)

        clear_t, hazy_t = self.to_tensor(clear), self.to_tensor(hazy)
        if self.density_aug:
            hazy_t = augment_haze_density(clear_t, hazy_t, self.density_alpha_range)

        return hazy_t, clear_t
