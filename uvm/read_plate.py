"""
Point this at ANY photo -- a tight plate crop or a full traffic scene with
several vehicles -- and get back predicted plate text for every plate in
it. Runs the project's own trained YOLOv11 plate detector first (see
yolo_plate_detection/), crops each detection, then dehaze-then-OCR on
each crop (same pipeline as evaluate_pipeline_ocr.py). If detection finds
nothing (e.g. the image is already a tight plate crop with no surrounding
scene for YOLO to key off of), it falls back to treating the whole image
as a single plate -- so this still works on the dataset's own pre-cropped
images without needing a flag.

--image_dir processes every image in a folder in one run (models loaded
once, not once per image) -- for a whole batch of your own test photos.

Run from WSL2 (needs the real mamba_ssm CUDA kernel for the UVMNet path):
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python read_plate.py --image /path/to/your/photo.jpg
    python read_plate.py --image_dir /path/to/folder --save_annotated_dir out/
"""
import argparse
import os
import re

import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms

import test_uvmnet as uvm
from model_uvmnet import UVMNet
from model_lprnet import LPRNet
from clahe_preprocess import apply_clahe_pil
from dispatch_dehaze import local_contrast, DEFAULT_CONTRAST_THRESHOLD
from dataset_lprnet import INPUT_SIZE as LPRNET_INPUT_SIZE
from lprnet_vocab import ctc_greedy_decode
from plate_format import ctc_format_constrained_decode
from two_line_plate import flatten_for_mode, normalize_plate_layout

# Crops with width/height in this range are read both as single-line and flattened
# two-line (see read_plate); outside it the plain aspect-ratio rule is reliable.
AMBIGUOUS_ASPECT = (1.9, 2.6)

DEFAULT_YOLO_CKPT = os.path.join(
    "..", "yolo_plate_detection", "runs", "detect", "runs", "detect",
    "anpr_plate_model_15k", "weights", "best.pt",
)


def _natural_sort_key(fname):
    """Splits a filename into text/number chunks so plain sorted() doesn't
    put '10.jpg' before '2.jpg' (lexicographic '1' < '2' as characters) --
    this orders them numerically instead: 1, 2, 3, ..., 10, 11, ..."""
    return [int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", fname)]


def dehaze(uvmnet, image: Image.Image, device):
    """Same routing as dispatch_dehaze.py -- returns (dehazed_image, method_name, contrast)."""
    c = local_contrast(image)
    force = os.environ.get("DEHAZE_METHOD", "")        # "uvm" / "clahe": skip the contrast routing (for comparing methods)
    if force == "clahe" or (force != "uvm" and c < DEFAULT_CONTRAST_THRESHOLD):
        return apply_clahe_pil(image), "CLAHE", c
    return uvm.process_image(uvmnet, image, device), "UVMNet", c


@torch.no_grad()
def read_plate(lprnet, image: Image.Image, device):
    """Runs LPRNet once and returns (raw_text, format_constrained_text).
    Unlike evaluate_pipeline_ocr.py's read_plate() (string-only, kept
    there for the eval script's own simplicity), this keeps the raw
    log_probs around so the format constraint can consult the network's
    OWN per-position probabilities (ctc_format_constrained_decode) rather
    than a fixed lookup-table swap -- see plate_format.py's docstring for
    why that matters (recovers characters like 'Q' a fixed table can't).
    Also applies normalize_plate_layout() first -- see two_line_plate.py:
    a two-line plate (roughly square crop) gets flattened to one wide
    row before resizing, instead of being vertically squashed into noise."""
    image = image.convert("RGB")
    low, high = AMBIGUOUS_ASPECT
    if low <= image.width / image.height < high:
        # Aspect ratio can't tell a short single-line plate from a wide two-line one
        # here (e.g. 122x55 single-line, 92x74 two-line): read both ways and keep the
        # reading that looks like a plate, then the one the network is more sure of.
        candidates = [image, flatten_for_mode(image)]
    else:
        candidates = [normalize_plate_layout(image)]
    best = None
    for cand in candidates:
        raw_text, fmt_text, conf = _decode(lprnet, cand, device)
        key = (looks_like_plate(fmt_text), conf)
        if best is None or key > best[0]:
            best = (key, raw_text, fmt_text)
    return best[1], best[2]


def _decode(lprnet, img, device):
    """One LPRNet pass on an already layout-normalised crop. Returns
    (raw_text, format_constrained_text, mean max log-prob per timestep)."""
    img_t = transforms.ToTensor()(img.resize(LPRNET_INPUT_SIZE)).unsqueeze(0).to(device)
    log_probs = torch.log_softmax(lprnet(img_t), dim=1).permute(2, 0, 1).cpu()  # (T, 1, C)
    return (ctc_greedy_decode(log_probs)[0], ctc_format_constrained_decode(log_probs),
            log_probs[:, 0, :].max(dim=1).values.mean().item())


def read_one_plate(crop, lprnet, uvmnet, device, skip_dehaze):
    """Returns a dict with raw/dehazed predictions for a single plate crop,
    plus a format-constrained version of each (see plate_format.py) --
    only differs from the raw prediction when the length already matches
    a known Indian plate format and some position has the wrong
    letter/digit class for its slot."""
    result = {}
    result["pred_raw"], result["pred_raw_fmt"] = read_plate(lprnet, crop, device)
    if not skip_dehaze:
        dehazed_img, method, contrast = dehaze(uvmnet, crop, device)
        result["pred_dehazed"], result["pred_dehazed_fmt"] = read_plate(lprnet, dehazed_img, device)
        result["method"] = method
        result["contrast"] = contrast
        result["dehazed_img"] = dehazed_img
    return result


def _fmt_note(raw_text, fmt_text):
    """Small suffix noting the format-corrected reading when it differs
    from the raw one -- makes the O/0-style corrections visible instead
    of silently overriding what the network actually predicted."""
    if fmt_text == raw_text:
        return ""
    return f"  -> format-corrected: {fmt_text!r}"


def looks_like_plate(text):
    """Cheap plausibility check on a reading: Indian plates have >= 8 characters
    with at least 4 digits and 2 letters. Rejects the 2-5 character fragments the
    detector's low-confidence boxes produce on stickers, signs and bumper text."""
    return (len(text) >= 8 and sum(c.isdigit() for c in text) >= 4
            and sum(c.isalpha() for c in text) >= 2)


def process_image(img, name, yolo, lprnet, uvmnet, device, args, annotated_out_path=None, crops_out_dir=None):
    """Runs detection + dehaze-then-OCR on one already-loaded PIL image.
    Prints results and optionally saves an annotated copy / crops."""
    boxes = []
    if yolo is not None:
        results = yolo.predict(img, conf=args.yolo_conf, imgsz=args.yolo_imgsz, verbose=False)
        for r in results:
            for xyxy, conf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                boxes.append((tuple(xyxy), conf))

    print(f"\n=== {name} ===  ({len(boxes)} plate(s) detected)" if yolo is not None else f"\n=== {name} ===")

    if crops_out_dir:
        os.makedirs(crops_out_dir, exist_ok=True)

    if not boxes:
        if yolo is not None:
            print("  no plates detected -- falling back to treating the whole image as one plate crop.")
        result = read_one_plate(img, lprnet, uvmnet, device, args.skip_dehaze)
        print(f"  raw image  -> predicted: {result['pred_raw']!r}{_fmt_note(result['pred_raw'], result['pred_raw_fmt'])}")
        if not args.skip_dehaze:
            print(f"  dehazed    -> predicted: {result['pred_dehazed']!r}{_fmt_note(result['pred_dehazed'], result['pred_dehazed_fmt'])}  "
                  f"(method: {result['method']}, local_contrast: {result['contrast']:.1f})")
        return

    annotated = img.copy() if annotated_out_path else None
    draw = ImageDraw.Draw(annotated) if annotated else None

    # Read every detection first, then drop the ones whose reading doesn't look like a
    # plate at all (tiny stickers, signs, bumper text: 'RE5', 'WLM43', ...). They are
    # listed in one summary line, not silently discarded.
    reads = []
    for i, (box, conf) in enumerate(boxes):
        crop = img.crop(box)
        result = read_one_plate(crop, lprnet, uvmnet, device, args.skip_dehaze)
        final = result["pred_raw_fmt"] if args.skip_dehaze else result["pred_dehazed_fmt"]
        reads.append((i, box, conf, crop, result, final))
    shown = reads if args.keep_all else [r for r in reads if looks_like_plate(r[5])]
    if not shown and reads:  # nothing plausible: still show the most confident box, flagged
        shown = [max(reads, key=lambda r: r[2])]
        print("  (no detection read like a plate; showing the most confident one, probably unreliable)")
    hidden = [r for r in reads if not any(r is s for s in shown)]
    if hidden:
        print("  hidden as non-plates: " + ", ".join(
            f"#{r[0]} {r[5]!r} (conf {r[2]:.2f}, {r[3].width}x{r[3].height})" for r in hidden)
            + "  [--keep_all to show]")

    for i, box, conf, crop, result, _ in shown:
        print(f"  plate {i}  box={tuple(round(v) for v in box)}  conf={conf:.2f}  size={crop.size}")
        print(f"    raw image  -> predicted: {result['pred_raw']!r}{_fmt_note(result['pred_raw'], result['pred_raw_fmt'])}")
        label_text = result["pred_raw_fmt"]
        if not args.skip_dehaze:
            print(f"    dehazed    -> predicted: {result['pred_dehazed']!r}{_fmt_note(result['pred_dehazed'], result['pred_dehazed_fmt'])}  "
                  f"(method: {result['method']}, local_contrast: {result['contrast']:.1f})")
            label_text = result["pred_dehazed_fmt"]

        if crops_out_dir:
            crop.save(os.path.join(crops_out_dir, f"{os.path.splitext(name)[0]}_plate{i}.png"))

        if draw:
            draw.rectangle(box, outline=(0, 255, 0), width=2)
            draw.text((box[0], max(0, box[1] - 12)), label_text, fill=(0, 255, 0))

    if annotated:
        annotated.save(annotated_out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="Path to a single photo -- full scene or a tight plate crop.")
    ap.add_argument("--image_dir", default=None,
                     help="Path to a folder of photos -- processes every image in it, "
                          "models loaded once. Use instead of --image.")
    ap.add_argument("--uvmnet_ckpt", default="checkpoints_combined/uvmnet_combined_best_stateonly.pt")
    ap.add_argument("--lprnet_ckpt", default="checkpoints_lprnet/lprnet_avg.pt")
    ap.add_argument("--yolo_ckpt", default=DEFAULT_YOLO_CKPT,
                     help="Trained plate-detector weights. Set --no_detect to skip "
                          "detection entirely and treat each image as a single plate crop.")
    ap.add_argument("--yolo_conf", type=float, default=0.25)
    ap.add_argument("--yolo_imgsz", type=int, default=640,
                     help="Detector input size. 1280 finds many more plates in large photos (small plates).")
    ap.add_argument("--keep_all", action="store_true",
                     help="Show every detection, including ones whose reading doesn't look "
                          "like a plate (default: those are hidden and summarised in one line).")
    ap.add_argument("--no_detect", action="store_true",
                     help="Skip YOLO detection -- treat each image as one plate crop "
                          "(the old behavior, e.g. for the dataset's own pre-cropped images).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default="cuda", choices=["torch", "cuda"])
    ap.add_argument("--uvmb_size", type=int, default=32)
    ap.add_argument("--save_crops_dir", default=None,
                     help="Optional directory to save each detected plate crop.")
    ap.add_argument("--save_annotated", default=None,
                     help="With --image: path to save the annotated image. With --image_dir: "
                          "a directory -- one annotated file per input image.")
    ap.add_argument("--skip_dehaze", action="store_true",
                     help="Read straight into LPRNet, no dehazing at all -- useful to "
                          "compare against the dehazed prediction.")
    args = ap.parse_args()

    if not args.image and not args.image_dir:
        ap.error("pass either --image or --image_dir")

    device = torch.device(args.device)

    print("Loading LPRNet...")
    lprnet = LPRNet(num_classes=37).to(device)
    lprnet.load_state_dict(torch.load(args.lprnet_ckpt, map_location=device))
    lprnet.eval()

    uvmnet = None
    if not args.skip_dehaze:
        print("Loading UVMNet...")
        uvmnet = UVMNet(n_channels=3, backend=args.backend, uvmb_size=args.uvmb_size).to(device)
        uvmnet.load_state_dict(torch.load(args.uvmnet_ckpt, map_location=device))
        uvmnet.eval()

    yolo = None
    if not args.no_detect:
        from ultralytics import YOLO
        print("Loading YOLO plate detector...")
        yolo = YOLO(args.yolo_ckpt)

    if args.image:
        img = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")      # phone photos are often stored sideways
        process_image(img, os.path.basename(args.image), yolo, lprnet, uvmnet, device, args,
                      annotated_out_path=args.save_annotated, crops_out_dir=args.save_crops_dir)
    else:
        exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        files = sorted((f for f in os.listdir(args.image_dir) if f.lower().endswith(exts)), key=_natural_sort_key)
        print(f"Found {len(files)} image(s) in {args.image_dir}")
        if args.save_annotated:
            os.makedirs(args.save_annotated, exist_ok=True)
        for fname in files:
            img = ImageOps.exif_transpose(Image.open(os.path.join(args.image_dir, fname))).convert("RGB")
            ann_path = os.path.join(args.save_annotated, fname) if args.save_annotated else None
            process_image(img, fname, yolo, lprnet, uvmnet, device, args,
                          annotated_out_path=ann_path, crops_out_dir=args.save_crops_dir)


if __name__ == "__main__":
    main()
