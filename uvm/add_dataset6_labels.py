"""
One-off helper: crops the known plate boxes out of your dataset6 photos
(same boxes YOLOv11 already detected) and saves them into
datasets/dataset/clear/ as new scene images, then prints a JSON snippet
to paste into evaluation/baseline_ocr/ground_truth.json.

This does NOT guess the plate text for you -- the TEXTS dict below is
pre-filled with my best reconstruction from this conversation (combining
each raw OCR prediction with the specific character corrections you
pointed out), but you should check every value against the actual photo
before saving, the same way you caught the Q and V errors. Edit the
TEXTS dict directly if anything is wrong, then re-run.

Run from WSL2:
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python add_dataset6_labels.py
"""
import json
import os

from PIL import Image

DATASET6_DIR = "../datasets/dataset6"
CLEAR_DIR = "../datasets/dataset/clear"
GROUND_TRUTH_PATH = "../evaluation/baseline_ocr/ground_truth.json"

# (source image, box, new scene_id, best-guess text -- VERIFY before trusting)
ENTRIES = [
    ("20260929_170316.jpg", (328, 2183, 761, 2299), "ds6_316_p0", "KL21AA2696"),   # verified exact match
    ("20260929_170316.jpg", (2351, 214, 2658, 292), "ds6_316_p1", "KL01BN0345"),   # verified, consistent reads
    ("20260929_170320.jpg", (750, 1372, 1217, 1492), "ds6_320_p0", "KL01BN0345"),  # same plate, different photo
    ("20260929_170303.jpg", (1160, 1652, 1622, 1771), "ds6_303_p0", "KL19Q9369"),  # Q confirmed by you -- CHECK REST
    ("WhatsApp Image 2026-09-30 at 9.55.19 AM.jpeg", (464, 818, 716, 926), "ds6_9551_p0", "KL22V3025"),  # V confirmed by you -- CHECK REST
    ("WhatsApp Image 2026-09-30 at 9.55.20 AM.jpeg", (708, 587, 813, 617), "ds6_9552_p0", "KL01DK1675"),  # UNVERIFIED -- never checked against photo
]


def main():
    os.makedirs(CLEAR_DIR, exist_ok=True)

    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)

    new_entries = {}
    for fname, box, scene_id, text in ENTRIES:
        src = os.path.join(DATASET6_DIR, fname)
        img = Image.open(src).convert("RGB")
        crop = img.crop(box)
        out_path = os.path.join(CLEAR_DIR, scene_id + ".jpg")
        crop.save(out_path)
        new_entries[scene_id] = text
        print(f"saved {out_path}  size={crop.size}  text={text!r}")

    print("\nPaste/merge this into ground_truth.json (after checking every text value):")
    print(json.dumps(new_entries, indent=2))

    already_present = [k for k in new_entries if k in gt]
    if already_present:
        print(f"\nNOTE: these scene_ids already exist in ground_truth.json, won't overwrite: {already_present}")


if __name__ == "__main__":
    main()
