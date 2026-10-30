"""
Scores plate detectors on hand-corrected traffic frames (e.g. ../datasets/det_test_frames, made with
make_det_frames.py and corrected in makesense.ai into labels_fixed/).

For each setup (checkpoint + detection size(s)) it reports, at IoU >= 0.5:
  recall      share of labelled plates found (overall and by plate width in pixels)
  precision   share of boxes that are real plates, at the operating confidence (--conf)
  AP50        area under the precision-recall curve over all confidences
Several sizes (e.g. 640 1280) are combined the way read_video.py does it: the first size's boxes are
kept and later sizes only add boxes that don't overlap them.

    python eval_detector.py --setups "best.pt:640" "best.pt:1280" "best.pt:640,1280"
"""
import argparse
import os

from PIL import Image
from ultralytics import YOLO

DEFAULT_CKPT = "runs/detect/runs/detect/anpr_plate_model_15k/weights/best.pt"
BINS = [(0, 40, "<40px"), (40, 80, "40-80px"), (80, 10 ** 6, ">=80px")]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


def load_gt(path, W, H):
    out = []
    for l in open(path):
        s = l.split()
        if len(s) == 5:
            cx, cy, w, h = map(float, s[1:])
            out.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    return out


def detect(model, img, sizes, conf):
    dets = []
    for k, sz in enumerate(sizes):
        for r in model.predict(img, conf=conf, imgsz=sz, verbose=False):
            for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                if k > 0 and any(iou(b, d) > 0.3 for d, _ in dets):
                    continue
                dets.append((tuple(b), c))
    return dets


def evaluate(frames, model, sizes, conf_op, min_iou=0.5):
    """frames: [(PIL image, [gt boxes])]. Detections collected at a low threshold so AP can be computed."""
    all_dets, n_gt = [], 0
    gt_found_op = {b: 0 for _, _, lo_hi in [(0, 0, b) for b in BINS] for b in [lo_hi[2]]}
    gt_total = {b[2]: 0 for b in BINS}
    tp_op = fp_op = 0
    for img, gts in frames:
        n_gt += len(gts)
        for g in gts:
            gt_total[next(n for lo, hi, n in BINS if lo <= g[2] - g[0] < hi)] += 1
        dets = sorted(detect(model, img, sizes, 0.05), key=lambda d: -d[1])
        used = set()
        for box, c in dets:                     # greedy matching, highest confidence first
            best, j = 0.0, None
            for k, g in enumerate(gts):
                if k not in used:
                    v = iou(box, g)
                    if v > best:
                        best, j = v, k
            hit = best >= min_iou
            if hit:
                used.add(j)
            all_dets.append((c, hit))
            if c >= conf_op:
                if hit:
                    tp_op += 1
                    g = gts[j]
                    gt_found_op[next(n for lo, hi, n in BINS if lo <= g[2] - g[0] < hi)] += 1
                else:
                    fp_op += 1
    # AP50: precision-recall over all detections sorted by confidence (VOC-style, all points)
    all_dets.sort(key=lambda d: -d[0])
    tp = fp = 0
    prec, rec = [], []
    for c, hit in all_dets:
        tp += hit
        fp += not hit
        prec.append(tp / (tp + fp))
        rec.append(tp / max(n_gt, 1))
    ap, prev_r = 0.0, 0.0
    for i in range(len(rec)):
        p_interp = max(prec[i:]) if prec[i:] else 0
        ap += (rec[i] - prev_r) * p_interp
        prev_r = rec[i]
    return {
        "recall": tp_op / max(n_gt, 1),
        "precision": tp_op / max(tp_op + fp_op, 1),
        "false_boxes": fp_op,
        "ap50": ap,
        "by_size": {n: (gt_found_op[n], gt_total[n]) for _, _, n in BINS},
        "n_gt": n_gt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="../datasets/det_test_frames")
    ap.add_argument("--labels", default="labels_fixed")
    ap.add_argument("--setups", nargs="+", default=[f"{DEFAULT_CKPT}:640", f"{DEFAULT_CKPT}:1280",
                                                    f"{DEFAULT_CKPT}:640,1280"],
                    help="checkpoint:size[,size...]")
    ap.add_argument("--conf", type=float, default=0.3, help="operating confidence (read_video.py uses 0.3)")
    ap.add_argument("--iou", type=float, default=0.5, help="box overlap needed to count as found")
    args = ap.parse_args()

    frames = []
    lab_dir = os.path.join(args.frames, args.labels)
    for n in sorted(os.listdir(lab_dir)):
        if not n.endswith(".txt") or n == "labels.txt":
            continue
        img = Image.open(os.path.join(args.frames, "images", n[:-4] + ".jpg")).convert("RGB")
        frames.append((img, load_gt(os.path.join(lab_dir, n), *img.size)))
    n_gt = sum(len(g) for _, g in frames)
    print(f"{len(frames)} frames, {n_gt} labelled plates | operating confidence {args.conf}, IoU >= {args.iou}\n")
    print(f"{'setup':<44} {'recall':>7} {'precision':>9} {'false boxes':>11} {'AP50':>6}   recall by plate width")
    models = {}
    for s in args.setups:
        ck, sizes = s.rsplit(":", 1)
        sizes = [int(x) for x in sizes.split(",")]
        models.setdefault(ck, YOLO(ck))
        r = evaluate(frames, models[ck], sizes, args.conf, args.iou)
        by = "  ".join(f"{n} {f}/{t}" for n, (f, t) in r["by_size"].items())
        name = f"{os.path.basename(os.path.dirname(os.path.dirname(ck))) or ck} @ {'+'.join(map(str, sizes))}"
        print(f"{name:<44} {r['recall']:>7.1%} {r['precision']:>9.1%} {r['false_boxes']:>11} {r['ap50']:>6.3f}   {by}")


if __name__ == "__main__":
    main()
