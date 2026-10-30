"""
Paired (plate image, text) dataset for LPRNet, built from this project's
own labeled data: evaluation/baseline_ocr/ground_truth.json (218 manually
labeled scenes) matched against datasets/dataset/clear/<id>.jpg and every
datasets/dataset/hazy/<id>_vN.jpg variant.

Key point: haze doesn't change what characters are on the plate, so every
hazy variant of a labeled scene is a valid (image, text) pair sharing
that scene's ground truth -- 218 labeled scenes -> ~1,930 actual training
pairs once every variant is counted, not just 218.

Train/val split is done by scene id (not by individual variant), same
reasoning as every other dataset class in this project: avoids leaking
near-duplicate haze variants of the same plate across the split, which
would make validation accuracy look better than it really is.
"""
import os
import random

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from degrade import degrade_crop
from lprnet_vocab import encode
from two_line_plate import is_two_line, normalize_plate_layout

INPUT_SIZE = (200, 24)  # (W, H) -- must match model_lprnet.py's expected input


class LPRNetDataset(Dataset):
    def __init__(self, root, ground_truth, scene_ids=None, augment=True, input_size=INPUT_SIZE, degrade_prob=0.0,
                 degrade_before_flatten=False, weather_prob=0.0):
        """
        root: datasets/dataset (must contain clear/ and hazy/)
        ground_truth: dict {scene_id: plate_text}, already filtered to
            this split's scene_ids by the caller if scene_ids is given
        """
        self.clear_dir = os.path.join(root, "clear")
        self.hazy_dir = os.path.join(root, "hazy")
        self.augment = augment
        self.input_size = input_size
        self.degrade_prob = degrade_prob  # chance of degrade.degrade_crop() per training sample
        self.degrade_before_flatten = degrade_before_flatten
        self.weather_prob = weather_prob  # chance of rain+fog+grain (weather_crop.py) per training sample
        self.pairs = self._index_pairs(ground_truth, scene_ids)
        if not self.pairs:
            raise RuntimeError(f"No labeled image/text pairs found under {root!r}.")
        self.to_tensor = transforms.ToTensor()

    def _index_pairs(self, ground_truth, scene_ids):
        pairs = []
        hazy_files = os.listdir(self.hazy_dir)
        for scene_id, text in ground_truth.items():
            if scene_ids is not None and scene_id not in scene_ids:
                continue
            clear_path = os.path.join(self.clear_dir, scene_id + ".jpg")
            if os.path.exists(clear_path):
                pairs.append((clear_path, text))
            for f in hazy_files:
                if f.startswith(scene_id + "_v"):
                    pairs.append((os.path.join(self.hazy_dir, f), text))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def _augment(self, img: Image.Image) -> Image.Image:
        # Mild, plate-appropriate augmentation -- nothing that would
        # change which characters are legible, just camera/lighting
        # variation (brightness/contrast jitter, small rotation).
        if random.random() < 0.5:
            img = transforms.functional.adjust_brightness(img, random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            img = transforms.functional.adjust_contrast(img, random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            img = img.rotate(random.uniform(-5, 5), fillcolor=(128, 128, 128), expand=False)
        return img

    def __getitem__(self, idx):
        path, text = self.pairs[idx]
        img = Image.open(path).convert("RGB")
        if self.augment and self.weather_prob > 0 and random.random() < self.weather_prob:
            from weather_crop import weather_degrade
            img = weather_degrade(img)
        degrade = self.augment and self.degrade_prob > 0 and random.random() < self.degrade_prob
        if degrade and self.degrade_before_flatten:
            # Shrink/blur/compress BEFORE flattening: a small two-line plate in a video is small
            # as a whole (bike plates are ~30x20 px), so each of its two lines ends up only
            # ~8-15 px tall once flattened. Two-line crops therefore get a smaller height range.
            # (Tried as checkpoints_compare_heavy3: +6-8 pts on small two-line plates but worse on
            # clear plates, so it is opt-in.)
            two = is_two_line(img)
            img = degrade_crop(img, min_h=18 if two else 12, max_h=40 if two else 30)
        img = normalize_plate_layout(img)  # flattens two-line plates, see two_line_plate.py
        if degrade and not self.degrade_before_flatten:
            img = degrade_crop(img, min_w=32)  # recipe of the installed default model
        img = img.resize(self.input_size)

        if self.augment:
            img = self._augment(img)

        img_t = self.to_tensor(img)
        label = encode(text)
        return img_t, label, text


def ctc_collate(batch):
    """CTCLoss (default, non-batch_first target mode) wants: a padded/
    stacked input batch, ONE concatenated 1D tensor of every sample's
    target indices back to back (not padded per-sample), a length per
    input sequence, and a length per target. This collate builds exactly
    that instead of the default PyTorch collate (which would try to
    stack variable-length label lists into a tensor and fail)."""
    import torch

    imgs, labels, texts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.tensor([idx for label in labels for idx in label], dtype=torch.long)
    return imgs, targets, target_lengths, texts
