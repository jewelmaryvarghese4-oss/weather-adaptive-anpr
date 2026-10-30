"""
Paired clear/hazy dataset for NH-HAZE (ETH Zurich / NTIRE 2020) -- 55 real
outdoor scenes, non-homogeneous haze from a professional haze generator
(not synthetic), paired against a haze-free ground truth shot of the same
scene. Small, but the only real-haze source in this project's training mix
that isn't close-up vehicle/plate photos -- included specifically to widen
generalization to general outdoor real fog (e.g. dense highway fog), which
neither OTS (synthetic haze) nor FoggyPlates (real haze, but close-up
vehicles) covers well.

Layout (flat, matches the official zip / HF mirror exactly, no subfolders):
    <root>/<id>_GT.png
    <root>/<id>_hazy.png
"""
import os
import random

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from clahe_preprocess import apply_clahe_pil

GT_SUFFIX = "_GT.png"
HAZY_SUFFIX = "_hazy.png"


class PairedNHHazeDataset(Dataset):
    def __init__(self, root, patch_size=256, augment=True, ids=None, limit=None, clahe=False):
        self.root = root
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

    def _index_pairs(self, ids):
        pairs = []
        for f in sorted(os.listdir(self.root)):
            if not f.endswith(HAZY_SUFFIX):
                continue
            scene_id = f[: -len(HAZY_SUFFIX)]
            if ids is not None and scene_id not in ids:
                continue
            gt_path = os.path.join(self.root, scene_id + GT_SUFFIX)
            if os.path.exists(gt_path):
                pairs.append((gt_path, os.path.join(self.root, f)))
        return pairs

    @staticmethod
    def all_ids(root):
        ids = []
        for f in sorted(os.listdir(root)):
            if f.endswith(HAZY_SUFFIX):
                ids.append(f[: -len(HAZY_SUFFIX)])
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
        if self.clahe:
            hazy = apply_clahe_pil(hazy)

        clear, hazy = self._sync_crop(clear, hazy)

        if self.augment and random.random() < 0.5:
            clear = clear.transpose(Image.FLIP_LEFT_RIGHT)
            hazy = hazy.transpose(Image.FLIP_LEFT_RIGHT)

        return self.to_tensor(hazy), self.to_tensor(clear)
