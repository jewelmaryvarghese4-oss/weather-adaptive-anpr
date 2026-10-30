"""
Paired clear/hazy dataset for the Mendeley "Foggy-Hazy License Plates
Images" set (1001 real vehicle photos, Jaipur/Delhi/Gurugram).

Layout (matches the two Mendeley zips once each is extracted flat, no
per-image subfolders):
    <root>/clear/<id>.jpg   (ground truth; source zip mixes .jpg/.JPG case)
    <root>/hazy/<id>.png

One hazy image per clear image (1:1, unlike dataset_ots.py's many-variants-
per-scene), matched purely by the numeric id in the filename stem, case-
insensitive on extension.

These are full vehicle photos (not the tiny ANPR crops in datasets/dataset),
so this takes a synchronized random crop like dataset_ots.py /
dataset_traffic.py rather than resizing like dataset_plates.py.
"""
import os
import random

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from clahe_preprocess import apply_clahe_pil

CLEAR_EXTS = (".jpg", ".jpeg", ".png")


class PairedFoggyPlatesDataset(Dataset):
    def __init__(self, root, patch_size=256, augment=True, ids=None, limit=None, clahe=False):
        self.clear_dir = os.path.join(root, "clear")
        self.hazy_dir = os.path.join(root, "hazy")
        self.patch_size = patch_size
        self.augment = augment
        self.clahe = clahe
        self.pairs = self._index_pairs(ids)
        if not self.pairs:
            raise RuntimeError(f"No clear/hazy pairs found under {root!r}.")
        if limit is not None:
            rng = random.Random(0)
            rng.shuffle(self.pairs)
            self.pairs = self.pairs[:limit]
        self.to_tensor = transforms.ToTensor()

    def _index_clear(self):
        by_id = {}
        for f in os.listdir(self.clear_dir):
            stem, ext = os.path.splitext(f)
            if ext.lower() in CLEAR_EXTS:
                by_id[stem] = os.path.join(self.clear_dir, f)
        return by_id

    def _index_pairs(self, ids):
        clear_by_id = self._index_clear()
        pairs = []
        for f in sorted(os.listdir(self.hazy_dir)):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in CLEAR_EXTS:
                continue
            if ids is not None and stem not in ids:
                continue
            clear_path = clear_by_id.get(stem)
            if clear_path is not None:
                pairs.append((clear_path, os.path.join(self.hazy_dir, f)))
        return pairs

    @staticmethod
    def all_ids(root):
        clear_dir = os.path.join(root, "clear")
        ids = []
        for f in sorted(os.listdir(clear_dir)):
            stem, ext = os.path.splitext(f)
            if ext.lower() in CLEAR_EXTS:
                ids.append(stem)
        return ids

    def __len__(self):
        return len(self.pairs)

    def _sync_crop(self, clear, hazy):
        w, h = clear.size
        ps = self.patch_size
        if w < ps or h < ps:
            new_w, new_h = max(w, ps), max(h, ps)
            clear = clear.resize((new_w, new_h))
            hazy = hazy.resize((new_w, new_h))
            w, h = new_w, new_h

        x0 = random.randint(0, w - ps)
        y0 = random.randint(0, h - ps)
        box = (x0, y0, x0 + ps, y0 + ps)
        return clear.crop(box), hazy.crop(box)

    def __getitem__(self, idx):
        clear_path, hazy_path = self.pairs[idx]
        clear = Image.open(clear_path).convert("RGB")
        hazy = Image.open(hazy_path).convert("RGB")

        # Hazy PNGs and clear JPEGs here aren't always pixel-aligned in
        # size (source dataset didn't guarantee it) -- resize hazy to the
        # clear image's size before cropping so both crops sample the same
        # region.
        if hazy.size != clear.size:
            hazy = hazy.resize(clear.size)
        if self.clahe:
            hazy = apply_clahe_pil(hazy)

        clear, hazy = self._sync_crop(clear, hazy)

        if self.augment and random.random() < 0.5:
            clear = clear.transpose(Image.FLIP_LEFT_RIGHT)
            hazy = hazy.transpose(Image.FLIP_LEFT_RIGHT)

        return self.to_tensor(hazy), self.to_tensor(clear)
