"""
dataset.py
----------
Loads your existing foggy images only — no clean pairs, no synthetic fog
generation. Point FOGGY_DIR at the folder of real foggy images you already
collected from the ANPR camera.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class FoggyImageDataset(Dataset):
    def __init__(self, foggy_dir, crop_size=256):
        self.paths = [
            os.path.join(foggy_dir, f)
            for f in sorted(os.listdir(foggy_dir))
            if f.lower().endswith(IMG_EXTS)
        ]
        if len(self.paths) == 0:
            raise ValueError(f"No images found in {foggy_dir}")

        self.transform = T.Compose([
            T.RandomResizedCrop(crop_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),  # -> [0,1], shape (3,H,W)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)