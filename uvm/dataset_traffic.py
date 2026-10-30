"""
Paired clear/foggy dataset for the UVM_TRAFFIC_DATASET (Foggy Cityscapes).

Layout expected (matches the extracted Cityscapes zips exactly):
    <clear_dir>/<city>/<city>_XXXXXX_XXXXXX_leftImg8bit.png
    <hazy_dir>/<city>/<city>_XXXXXX_XXXXXX_leftImg8bit_foggy_beta_<b>.png

e.g. clear_dir=UVM_TRAFFIC_DATASET/leftImg8bit/train,
     hazy_dir=UVM_TRAFFIC_DATASET/leftImg8bit_foggy/train.

Each clear scene has up to 3 synthetic fog variants (beta=0.005/0.01/0.02,
light->dense). All three are indexed as separate (hazy, clear) pairs, so
one epoch sees each scene's full fog-density curriculum, not just one
random draw of it.

Images are 2048x1024 -- too large to fit whole on an 8GB card at any
useful batch size, so __getitem__ takes a synchronized random crop (same
offset applied to both hazy and clear) instead of resizing, which would
throw away the fine detail dehazing needs to learn from.
"""
import os
import random

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

BETAS = ("0.005", "0.01", "0.02")
CLEAR_SUFFIX = "_leftImg8bit.png"


class PairedTrafficDataset(Dataset):
    def __init__(self, clear_dir, hazy_dir, patch_size=384, augment=True,
                 betas=BETAS, limit=None):
        self.patch_size = patch_size
        self.augment = augment
        self.pairs = self._index_pairs(clear_dir, hazy_dir, betas)
        if not self.pairs:
            raise RuntimeError(
                f"No clear/foggy pairs found under {clear_dir!r} / {hazy_dir!r}. "
                "Check that both dirs use the Cityscapes <city>/<file> layout."
            )
        if limit is not None:
            rng = random.Random(0)
            rng.shuffle(self.pairs)
            self.pairs = self.pairs[:limit]
        self.to_tensor = transforms.ToTensor()

    def _index_pairs(self, clear_dir, hazy_dir, betas):
        pairs = []
        for city in sorted(os.listdir(clear_dir)):
            city_clear = os.path.join(clear_dir, city)
            city_hazy = os.path.join(hazy_dir, city)
            if not os.path.isdir(city_clear) or not os.path.isdir(city_hazy):
                continue
            for fname in sorted(os.listdir(city_clear)):
                if not fname.endswith(CLEAR_SUFFIX):
                    continue
                stem = fname[: -len(CLEAR_SUFFIX)]
                clear_path = os.path.join(city_clear, fname)
                for beta in betas:
                    hazy_path = os.path.join(
                        city_hazy, f"{stem}_leftImg8bit_foggy_beta_{beta}.png"
                    )
                    if os.path.exists(hazy_path):
                        pairs.append((clear_path, hazy_path))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def _sync_crop(self, clear, hazy):
        w, h = clear.size
        ps = self.patch_size
        if w < ps or h < ps:
            # Guard only -- Cityscapes images are always 2048x1024, so this
            # shouldn't trigger with any sane patch_size.
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

        clear, hazy = self._sync_crop(clear, hazy)

        if self.augment and random.random() < 0.5:
            clear = clear.transpose(Image.FLIP_LEFT_RIGHT)
            hazy = hazy.transpose(Image.FLIP_LEFT_RIGHT)

        # Matches train_uvmnet.py's existing convention: (hazy, clear).
        return self.to_tensor(hazy), self.to_tensor(clear)
