"""
On-the-fly synthetic license plate generator for LPRNet pretraining.

Why: the real labeled dataset (dataset_lprnet.py) has only ~197 distinct
plate TEXTS in its training split -- nowhere near enough character/text
diversity for a network to learn general character recognition from
scratch (it just memorizes those 197 plates instead, confirmed by the
first real training run: train loss -> 0.32, val_cer stuck ~0.70).

This generates a fresh random plate image every call -- random text,
random length, random rendering choices -- so across a pretraining run
the network sees effectively unlimited distinct text, forcing it to
learn actual character shapes instead of memorizing a small fixed set.
It's rendered, not real, so it can't teach real-world camera/lighting
quirks -- that's what the fine-tuning stage on real data (train_lprnet.py
with --init_ckpt pointing at this pretrained checkpoint) is for.

Nothing here is saved to disk -- generation happens in __getitem__, so
every epoch (and even every sample within an epoch) is different for the
(unseeded) training set. The optional `seed` gives a REPRODUCIBLE dataset
instead (same plate at the same index every time) -- used for the
validation set, so epoch-to-epoch comparisons are meaningful instead of
comparing against a different random val set each time.
"""
import os
import random

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from lprnet_vocab import ALPHABET, encode

INPUT_SIZE = (200, 24)  # (W, H) -- must match model_lprnet.py / dataset_lprnet.py

# Render at a larger size than the final input, then downscale -- crisper
# text edges than drawing directly at 200x24, and downscaling itself acts
# as a mild, realistic anti-aliasing/blur.
RENDER_SIZE = (800, 96)

_FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/arialbd.ttf",
    "/mnt/c/Windows/Fonts/courbd.ttf",
    "/mnt/c/Windows/Fonts/consolab.ttf",
    "/mnt/c/Windows/Fonts/verdanab.ttf",
    "/mnt/c/Windows/Fonts/tahomabd.ttf",
    "/mnt/c/Windows/Fonts/calibrib.ttf",
    "/mnt/c/Windows/Fonts/trebucbd.ttf",
    "/mnt/c/Windows/Fonts/georgiab.ttf",
    "/mnt/c/Windows/Fonts/segoeuib.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _find_fonts():
    """Returns every candidate font that actually exists, not just the
    first -- the whole point is picking a DIFFERENT font per sample (see
    _render_plate), not a single fixed one. Previously this only ever
    returned the first match, so every synthetic plate across an entire
    pretraining run was rendered in exactly one font -- the network never
    had to learn what distinguishes similarly-shaped letters (e.g. Q vs
    O, V vs Y) across different renderings of the same character, which
    is exactly the kind of signal real-world font variety would provide.
    Confirmed as a real gap: a genuinely confusable pair (V misread as Y)
    showed up in testing, and the network's own class-restricted
    confidence didn't favor the correct letter even when the digit/letter
    class was forced correctly -- i.e. it hadn't learned the shape
    difference at all, not just mis-decoded it."""
    return [p for p in _FONT_CANDIDATES if os.path.exists(p)]


_FONT_PATHS = _find_fonts()

# Plate background/text color pairs -- a few common real-world schemes
# (white/light-gray plate with dark text, yellow commercial-style plate)
# so the network doesn't overfit to a single color scheme either.
_COLOR_SCHEMES = [
    ((235, 235, 235), (20, 20, 20)),   # light gray plate, near-black text
    ((250, 250, 245), (10, 10, 10)),   # off-white plate, black text
    ((240, 200, 40), (20, 20, 20)),    # yellow commercial plate, black text
    ((225, 225, 235), (30, 30, 60)),   # light blue-gray plate, dark blue text
    ((30, 120, 90), (255, 255, 255)),  # green EV plate (private), white text
    ((30, 120, 90), (240, 200, 40)),   # green EV plate (commercial), yellow text
]
# Red temporary / trade-certificate plates (e.g. 'TN 02 TC 0143', 'MH 15 TC 554'): white text on red. They were read as
# 'M0' / 'I15' because no red plate had ever been seen in training. Used for a small share of samples only.
_RED_SCHEMES = [((200, 45, 40), (255, 255, 255)), ((170, 30, 30), (245, 245, 245))]
_RED_PLATE_PROB = 0.07


# Characters that look alike once a plate is small/blurry. Placing them NEXT to each
# other (e.g. the 'B8' in 'AB8894') is the hard case: the network tends to merge or
# drop one of them.
_LOOKALIKES = ["B8", "8B", "O0", "0O", "Q0", "0Q", "D0", "0D", "S5", "5S", "Z2", "2Z", "G6", "6G", "I1", "1I",
               "U0", "V7", "7V"]


def _random_text(rng, min_len=4, max_len=11, heavy_repeats=False):
    """Uniform-random text across the full alphanumeric alphabet, same as
    the original generator -- NOT constrained to Indian plate format.

    An earlier version of this function forced every sample into strict
    Indian plate format (LLDDLLDDDD/LLDDLDDDD, see plate_format.py) to
    fix a CTC repeat-collapse bug (see below). That made things WORSE:
    checking evaluation/baseline_ocr/ground_truth.json's actual length
    distribution afterward showed only 27% of the real fine-tuning texts
    are even length 9 or 10 -- the rest are things like 'Y', 'PUI8BES',
    'CHIOOSE', 'LAMBO' (decals/partial crops/non-standard text, not
    strict plate format). Locking pretraining into a format most of the
    real data doesn't follow regressed val_cer from 0.2490 to 0.3418.
    Reverted to unconstrained random text to match that real diversity.

    Still deliberately forces a repeated adjacent character (same letter
    or same digit twice in a row) with 40% probability, independent of
    any format assumption -- this part of the fix stands on its own.
    Why: CTC's greedy decode collapses consecutive identical predicted
    characters unless the network predicts a blank between them, so it
    has to explicitly learn to separate doubled characters wherever they
    occur, not just in a specific plate slot. A real photo with plate
    'KL21AA2696' had its second 'A' dropped (predicted 'KL21A2696') --
    pure uniform-random generation only produces an adjacent repeat by
    chance (~1/36 per position pair), too rare for the network to have
    learned this reliably."""
    length = rng.randint(min_len, max_len)
    chars = [rng.choice(ALPHABET) for _ in range(length)]

    if length >= 2 and rng.random() < 0.4:
        i = rng.randrange(length - 1)
        chars[i + 1] = chars[i]

    # Triple runs (e.g. the '888' in DL1ZC8889) -- a real photo came back as 'DL1ZC889'
    # because pairs alone don't teach the network to separate THREE in a row.
    if length >= 3 and rng.random() < 0.15:
        i = rng.randrange(length - 2)
        chars[i + 1] = chars[i + 2] = chars[i]

    if heavy_repeats:
        # Extra repeats on top of the above, mostly in the DIGIT block at the end of a
        # plate (real plates: 'KL07DC7777', 'KL21AB8894'), plus look-alike neighbours.
        if length >= 5 and rng.random() < 0.35:                  # run of 2-4 identical digits near the end
            run = rng.choice([2, 3, 3, 4])
            if rng.random() < 0.6:   # usually in the final digit block ...
                start = rng.randint(max(0, length - 4 - 1), max(0, length - run))
            else:                    # ... but also anywhere (e.g. the '66' in KL66A7879)
                start = rng.randint(0, max(0, length - run))
            d = rng.choice("0123456789")
            for k in range(run):
                if start + k < length:
                    chars[start + k] = d
        if length >= 4 and rng.random() < 0.3:                   # a look-alike pair, e.g. B8
            pair = rng.choice(_LOOKALIKES)
            i = rng.randrange(length - 1)
            chars[i], chars[i + 1] = pair[0], pair[1]

    # 'Q' was almost never produced by the network on real plates (0.37 % of real training characters against 11 % for '0'),
    # so on plates like 'RJ14QC9780' it read a zero. Make sure the synthetic data shows it regularly.
    if length >= 4 and rng.random() < 0.12:
        chars[rng.randrange(length)] = "Q"

    return "".join(chars)


def _render_plate(rng, text: str) -> Image.Image:
    if rng.random() < _RED_PLATE_PROB:
        bg_color, text_color = rng.choice(_RED_SCHEMES)
    else:
        bg_color, text_color = rng.choice(_COLOR_SCHEMES)
    img = Image.new("RGB", RENDER_SIZE, bg_color)
    draw = ImageDraw.Draw(img)

    # Fit font size to the text length so short and long plates both use
    # the available width sensibly, same as real plates do.
    font_size = max(30, int(RENDER_SIZE[0] * 1.5 / max(len(text), 1)))
    font_size = min(font_size, 80)
    font_path = rng.choice(_FONT_PATHS) if _FONT_PATHS else None
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (RENDER_SIZE[0] - text_w) / 2 - bbox[0]
    y = (RENDER_SIZE[1] - text_h) / 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=text_color)

    # A plain rendered border around the plate, like the real crops have.
    draw.rectangle([2, 2, RENDER_SIZE[0] - 3, RENDER_SIZE[1] - 3], outline=text_color, width=3)

    return img


# ---- two-line plates (bikes, autos) ------------------------------------------------------
# Real two-line training plates are mostly 'KL 01 / AB 1234' (only state+district on top), so the
# model learned "the top line has 4 characters" and drops a series letter that sits next to the
# line break on plates like 'KL01C / V5776' or 'KL 01 / DL1983', and reads the old no-series
# 'KL 15 / 8194' as 'KL15B194'. These layouts are generated here.
_STATES = ["KL"] * 8 + ["TN", "KA", "MH", "DL", "AP", "TS", "GJ", "RJ", "UP", "HR", "PB", "WB", "GA", "PY"]
_SERIES_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"   # Indian series skip I and O
_TWO_LINE_LAYOUTS = [  # (name, weight)
    ("series_bottom", 3.0),   # KL 01 / AB 1234
    ("series_split", 3.0),    # KL 01 C / V 5776   (series letter at the end of the top line)
    ("series_top", 1.5),      # KL 01 DL / 1983
    ("no_series", 1.5),       # KL 15 / 8194
    ("short_number", 1.5),    # KL 01 / CJ.145
]


def _two_line_parts(rng):
    """Returns (top_tokens, bottom_tokens); the label is all tokens joined with no separators."""
    state = rng.choice(_STATES)
    rto = f"{rng.randint(1, 99):02d}"
    series = "".join(rng.choice(_SERIES_LETTERS) for _ in range(rng.choice([1, 2, 2])))
    number = "".join(rng.choice("0123456789") for _ in range(4))
    if rng.random() < 0.25:                      # repeated digits, e.g. 7777 / 8894
        i = rng.randrange(3)
        number = number[:i] + number[i] * 2 + number[i + 2:]
    layout = rng.choices([n for n, _ in _TWO_LINE_LAYOUTS], weights=[w for _, w in _TWO_LINE_LAYOUTS])[0]
    if layout == "series_split" and len(series) == 1:
        series += rng.choice(_SERIES_LETTERS)
    if layout == "series_bottom":
        return [state, rto], [series, number]
    if layout == "series_split":
        return [state, rto, series[0]], [series[1:], number]
    if layout == "series_top":
        return [state, rto, series], [number]
    if layout == "no_series":
        return [state, rto], [number]
    return [state, rto], [series, number[:3]]      # short_number


def _render_two_line(rng):
    top, bottom = _two_line_parts(rng)
    text = "".join(top + bottom)
    sep = lambda: rng.choice([" ", " ", ".", "-", "", "·"])
    top_s, bottom_s = sep().join(top), sep().join(bottom)
    if rng.random() < 0.3:                          # dots/hyphens at odd places, like 'KL.31' / 'J.7054'
        bottom_s = bottom_s.replace(" ", ".")

    W = 520
    H = int(W / rng.uniform(1.15, 1.9))             # two-line plates are roughly square to 2:1
    bg_color, text_color = rng.choice(_COLOR_SCHEMES)
    img = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(img)
    font_path = rng.choice(_FONT_PATHS) if _FONT_PATHS else None

    def fit(s, max_w, max_h):
        size = 120
        while size > 12:
            f = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
            b = draw.textbbox((0, 0), s, font=f)
            if b[2] - b[0] <= max_w and b[3] - b[1] <= max_h:
                return f, b
            size -= 4
        f = ImageFont.truetype(font_path, 12) if font_path else ImageFont.load_default()
        return f, draw.textbbox((0, 0), s, font=f)

    line_h = H * 0.40
    top_scale = rng.uniform(0.75, 1.0)              # top line is often smaller
    f_bot, b_bot = fit(bottom_s, W * 0.88, line_h)
    f_top, b_top = fit(top_s, W * 0.88 * top_scale, line_h * top_scale)
    y_top = H * rng.uniform(0.06, 0.14)
    y_bot = H * rng.uniform(0.52, 0.58)
    x_jit = lambda bw: (W - bw) / 2 + rng.uniform(-0.04, 0.04) * W
    draw.text((x_jit(b_top[2] - b_top[0]) - b_top[0], y_top - b_top[1]), top_s, font=f_top, fill=text_color)
    draw.text((x_jit(b_bot[2] - b_bot[0]) - b_bot[0], y_bot - b_bot[1]), bottom_s, font=f_bot, fill=text_color)
    draw.rectangle([3, 3, W - 4, H - 4], outline=text_color, width=4)
    if rng.random() < 0.4:                          # mounting bolts near the top corners
        r = int(H * 0.035)
        for cx in (int(W * 0.08), int(W * 0.92)):
            cy = int(H * 0.18)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(150, 150, 150), outline=text_color)
    return text, img


def _distort(rng, img: Image.Image) -> Image.Image:
    if rng.random() < 0.6:
        img = img.rotate(rng.uniform(-6, 6), fillcolor=(128, 128, 128), expand=False)
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.5)))
    if rng.random() < 0.7:
        img = transforms.functional.adjust_brightness(img, rng.uniform(0.6, 1.4))
    if rng.random() < 0.7:
        img = transforms.functional.adjust_contrast(img, rng.uniform(0.6, 1.4))
    if rng.random() < 0.4:
        img = transforms.functional.adjust_saturation(img, rng.uniform(0.5, 1.5))
    return img


class SyntheticPlateDataset(Dataset):
    def __init__(self, num_samples=30000, input_size=INPUT_SIZE, min_len=4, max_len=11, seed=None,
                 heavy_repeats=False, degrade_prob=0.0, two_line_frac=0.0):
        """num_samples: virtual epoch length -- generation is on-the-fly,
        so this just controls how many random plates count as 'one epoch'
        for the training loop/scheduler, not a fixed stored set.

        seed: None (default) -> every __getitem__ call uses the shared
        global `random` module, so both across epochs AND across workers
        every sample is freshly random (right for training). An int ->
        each index deterministically reproduces the same plate every
        time, via a fresh, disposable random.Random(seed + idx) created
        inside __getitem__ -- never stored as instance state, so this
        stays picklable for multiprocessing DataLoader workers (storing
        a stateful RNG on self was the earlier bug: worse, it stored the
        `random` MODULE itself when seed was None, which can't be
        pickled at all, crashing any run with --workers > 0)."""
        self.num_samples = num_samples
        self.input_size = input_size
        self.min_len = min_len
        self.max_len = max_len
        self.seed = seed
        self.heavy_repeats = heavy_repeats   # more repeated / look-alike characters, see _random_text
        self.degrade_prob = degrade_prob     # chance of degrade.degrade_crop() (small + blurry video look)
        self.two_line_frac = two_line_frac   # share of samples rendered as two-line Indian-layout plates
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = random.Random(self.seed + idx) if self.seed is not None else random

        if self.two_line_frac > 0 and rng.random() < self.two_line_frac:
            from two_line_plate import flatten_for_mode
            text, img = _render_two_line(rng)
            img = _distort(rng, img)
            if self.degrade_prob > 0 and rng.random() < self.degrade_prob:
                from degrade import degrade_crop
                img = degrade_crop(img, min_h=18, max_h=40)   # small as a whole, like a bike plate
            img = flatten_for_mode(img).resize(self.input_size)  # same flattening as real plates
            return self.to_tensor(img), encode(text), text

        text = _random_text(rng, self.min_len, self.max_len, self.heavy_repeats)
        img = _render_plate(rng, text)
        img = _distort(rng, img)
        if self.degrade_prob > 0 and rng.random() < self.degrade_prob:
            from degrade import degrade_crop
            img = degrade_crop(img)
        img = img.resize(self.input_size)
        img_t = self.to_tensor(img)
        label = encode(text)
        return img_t, label, text


if __name__ == "__main__":
    # Quick visual sanity check -- saves a few samples so you can eyeball
    # whether they look plate-like before spending pretraining time on them.
    ds = SyntheticPlateDataset(num_samples=8, seed=0)
    out_dir = "synthetic_plate_preview"
    os.makedirs(out_dir, exist_ok=True)
    for i in range(8):
        img_t, label, text = ds[i]
        img = transforms.ToPILImage()(img_t)
        img.save(os.path.join(out_dir, f"{i}_{text}.png"))
        print(f"{i}: text={text!r} label={label}")
    print(f"Saved previews to {out_dir}/ -- {len(_FONT_PATHS)} font(s) available: {_FONT_PATHS or 'none, using PIL default'}")

    # Reproducibility check: same seed+idx must give the same plate twice.
    ds2 = SyntheticPlateDataset(num_samples=8, seed=0)
    _, _, text_again = ds2[3]
    _, _, text_first = ds[3]
    assert text_again == text_first, "seeded dataset is not reproducible!"
    print(f"Reproducibility check passed (index 3 -> {text_first!r} both times).")
