"""
Paired clear/haze dataset for RESIDE-beta OTS (Outdoor Training Set).

Layout (matches the OTS_BETA Box mirror exactly, no reshaping needed):
    <root>/clear/<id>.jpg
    <root>/haze/<id>_<A>_<beta>.jpg   (multiple synthetic haze variants per clear id,
                                       A = atmospheric light, beta = scattering coeff)

e.g. haze/0001_0.9_0.16.jpg pairs with clear/0001.jpg -- the id is
everything before the first underscore in the haze filename.

Images are outdoor photos of varying resolution (not a fixed frame size
like Cityscapes), so this takes a synchronized random crop like
dataset_traffic.py rather than resizing like dataset_plates.py.
"""
import os
import random
import re

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from clahe_preprocess import apply_clahe_pil

ID_RE = re.compile(r"^([^_]+)_")


class PairedOTSDataset(Dataset):
    def __init__(self, root, patch_size=256, augment=True, clear_ids=None, limit=None, clahe=False):
        self.clear_dir = os.path.join(root, "clear")
        self.haze_dir = os.path.join(root, "haze")
        self.patch_size = patch_size
        self.augment = augment
        self.clahe = clahe
        self.pairs = self._index_pairs(clear_ids)
        if not self.pairs:
            raise RuntimeError(f"No clear/haze pairs found under {root!r}.")
        if limit is not None:
            rng = random.Random(0)
            rng.shuffle(self.pairs)
            self.pairs = self.pairs[:limit]
        self.to_tensor = transforms.ToTensor()

    def _index_pairs(self, clear_ids):
        # One os.listdir() for the lookup table, then pure in-memory dict
        # lookups per haze file -- NOT os.path.exists() per candidate.
        # WSL2's /mnt/c bridge does a slow round-trip per stat() call, and
        # OTS has ~70k haze files: one stat each was taking 10+ minutes of
        # pure I/O wait for something that should be instant.
        clear_by_id = {}
        for cf in os.listdir(self.clear_dir):
            cid, ext = os.path.splitext(cf)
            clear_by_id[cid] = os.path.join(self.clear_dir, cf)

        pairs = []
        for hf in sorted(os.listdir(self.haze_dir)):
            m = ID_RE.match(hf)
            if not m:
                continue
            cid = m.group(1)
            if clear_ids is not None and cid not in clear_ids:
                continue
            clear_path = clear_by_id.get(cid)
            if clear_path is None:
                continue
            pairs.append((clear_path, os.path.join(self.haze_dir, hf)))
        return pairs

    @staticmethod
    def all_clear_ids(root):
        clear_dir = os.path.join(root, "clear")
        ids = []
        for f in sorted(os.listdir(clear_dir)):
            ids.append(os.path.splitext(f)[0])
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
