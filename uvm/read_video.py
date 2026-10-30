"""
Reads plates in a VIDEO: detect plates in every frame (YOLO), follow each plate
across frames (simple IoU / distance tracker), read every crop with LPRNet, then
combine all readings of one track by character-level voting. One blurry frame no
longer decides the answer; the consensus over many frames does.

Outputs (in --out_dir):
  annotated.mp4          boxes + running consensus text per track
  track_<id>_best.png    the largest crop seen for each track
  tracks.csv             id, frames, consensus, plausible?, top raw readings

Run from WSL2:
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python read_video.py --video ../datasets/dataset13.mp4
"""
import argparse
import collections
import csv
import os
import re

import cv2
import torch
from PIL import Image
from ultralytics import YOLO

from lprnet_vocab import NUM_CLASSES
from model_lprnet import LPRNet
from read_plate import DEFAULT_YOLO_CKPT, looks_like_plate, read_plate


STRICT_PLATE = re.compile(r"^(DL\d{1,2}|[A-Z]{2}\d{2})[A-Z]{1,3}\d{3,4}$")
MIN_SUPPORT = 5            # frames that must produce the exact consensus text ...
MIN_SUPPORT_SHARE = 0.04   # ... and at least this share of the track's frames
STRICT_MIN_RATIO = 0.75  # picked from 3 tracks of dataset14 (0.81 right, 0.69 guess, 0.25 wrong); revisit with more data


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def center_dist(a, b):
    return (((a[0] + a[2]) - (b[0] + b[2])) ** 2 + ((a[1] + a[3]) - (b[1] + b[3])) ** 2) ** 0.5 / 2


class Track:
    def __init__(self, tid, box, frame_idx):
        self.id, self.box, self.last = tid, box, frame_idx
        self.readings = []      # (text, weight)
        self.best_crop, self.best_area = None, 0
        self.frames = 0
        self.log = []           # (frame, crop width, crop height, text) for --dump_readings

    def add(self, box, frame_idx, crop, text, weight):
        self.box, self.last = box, frame_idx
        self.frames += 1
        self.log.append((frame_idx, crop.width, crop.height, text))
        area = crop.width * crop.height
        if area > self.best_area:
            self.best_crop, self.best_area = crop, area
        if text:
            self.readings.append((text, weight))

    def consensus(self):
        """The 'medoid' reading: among readings that look like plates, the one with the
        smallest total edit distance to all the others (weighted by detection confidence x
        crop area). Unlike position-by-position voting it never mixes characters from
        readings of different lengths, and one wild reading can't drag the answer away."""
        # Prefer readings that fit a real plate layout (state + 2-digit district [Delhi: 1-2] +
        # 1-3 letters + 4 digits). This stops e.g. 'KL1Y8946' (district code with one digit)
        # out-voting the valid 'KL16Y8946'. But only when the layout-fitting reading is nearly as
        # common as the top plain plate-shaped one (>= STRICT_MIN_RATIO): otherwise it is probably a
        # reading that merely happens to fit (e.g. 'KL15B7884' from 4 frames vs 16 for 'KL15B7B4').
        strict = [(t, w) for t, w in self.readings if STRICT_PLATE.match(t)]
        good = [(t, w) for t, w in self.readings if looks_like_plate(t)]
        if strict and good:
            top_strict = collections.Counter(t for t, _ in strict).most_common(1)[0][1]
            top_good = collections.Counter(t for t, _ in good).most_common(1)[0][1]
            pool = strict if top_strict >= STRICT_MIN_RATIO * top_good else good
        else:
            pool = strict or good or self.readings
        if not pool:
            return "", False
        weight = collections.defaultdict(float)
        for t, w in pool:
            weight[t] += w
        best = min(weight, key=lambda c: (sum(wt * levenshtein(c, o) for o, wt in weight.items()), -weight[c]))
        return best, looks_like_plate(best)

    def support(self):
        """How many frames produced exactly the consensus reading (0 if no consensus)."""
        text, _ = self.consensus()
        return sum(1 for t, _ in self.readings if t == text) if text else 0

    def reliable(self):
        """A consensus counts as a real read only if enough frames AGREE on it (not just
        'it looks like a plate'): >= MIN_SUPPORT frames and >= MIN_SUPPORT_SHARE of the track."""
        text, good = self.consensus()
        sup = self.support()
        return bool(good) and sup >= MIN_SUPPORT and sup >= MIN_SUPPORT_SHARE * max(self.frames, 1)

    def top_readings(self, n=3):
        c = collections.Counter(t for t, _ in self.readings if t)
        return c.most_common(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_dir", default="../datasets/dataset13_out")
    ap.add_argument("--lprnet_ckpt", default="checkpoints_lprnet/lprnet_avg.pt")
    ap.add_argument("--yolo_ckpt", default=DEFAULT_YOLO_CKPT)
    ap.add_argument("--yolo_conf", type=float, default=0.3)
    ap.add_argument("--yolo_imgsz", type=int, nargs="+", default=[640, 1280],
                    help="size(s) YOLO resizes each frame to before detecting (640 = how it was trained). "
                         "Larger (1280) keeps small far-away plates bigger. With several sizes the first size's "
                         "boxes are kept and later sizes only ADD missed plates. Default 640+1280: on dataset14 "
                         "videos 3-4 it kept all 40 eye-checked plates 640 read right and added 4 more "
                         "(1280 alone: +7 but lost 4). Use --yolo_imgsz 640 for the faster old behaviour.")
    ap.add_argument("--stride", type=int, default=1, help="analyse every Nth frame (1 = all)")
    ap.add_argument("--min_w", type=int, default=24)
    ap.add_argument("--min_h", type=int, default=10)
    ap.add_argument("--min_frames", type=int, default=3, help="ignore tracks seen in fewer frames")
    ap.add_argument("--max_age", type=int, default=10, help="frames a track survives without a detection")
    ap.add_argument("--dump_readings", action="store_true",
                    help="also write readings.csv: every per-frame reading of every track")
    ap.add_argument("--save_track_crops", type=int, default=0,
                    help="save every Nth crop of every track to <out_dir>/track_crops/<id>/<frame>.png "
                         "(for building a benchmark that later models can be scored on). 0 = off")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    lprnet = LPRNet(num_classes=NUM_CLASSES).to(device)
    lprnet.load_state_dict(torch.load(args.lprnet_ckpt, map_location=device))
    lprnet.eval()
    yolo = YOLO(args.yolo_ckpt)

    os.makedirs(args.out_dir, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(os.path.join(args.out_dir, "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracks, active, next_id = [], [], 1
    shown = []  # (track, box) drawn on skipped frames too
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.stride == 0:
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dets = []
            for k, sz in enumerate(args.yolo_imgsz):
                for r in yolo.predict(rgb, conf=args.yolo_conf, imgsz=sz, verbose=False):
                    for xyxy, conf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                        box = tuple(xyxy)
                        if box[2] - box[0] < args.min_w or box[3] - box[1] < args.min_h:
                            continue
                        # extra sizes only ADD plates the first size missed; where both found the
                        # plate, the first size's box is kept (the reader was trained on those boxes)
                        if k > 0 and any(iou(box, b) > 0.3 for b, _ in dets):
                            continue
                        dets.append((box, conf))
            # greedy matching: best IoU first, fall back to nearest centre
            pairs = sorted(((iou(t.box, b), ti, di) for ti, t in enumerate(active)
                            for di, (b, _) in enumerate(dets)), reverse=True)
            used_t, used_d, match = set(), set(), {}
            for sc, ti, di in pairs:
                near = center_dist(active[ti].box, dets[di][0]) < 0.8 * max(
                    dets[di][0][2] - dets[di][0][0], dets[di][0][3] - dets[di][0][1])
                if ti in used_t or di in used_d or (sc < 0.1 and not near):
                    continue
                used_t.add(ti)
                used_d.add(di)
                match[di] = active[ti]
            for di, (box, conf) in enumerate(dets):
                tr = match.get(di)
                if tr is None:
                    tr = Track(next_id, box, idx)
                    next_id += 1
                    tracks.append(tr)
                    active.append(tr)
                crop = rgb.crop(tuple(int(v) for v in box))
                _, text = read_plate(lprnet, crop, device)
                tr.add(box, idx, crop, text, conf * crop.width * crop.height)
                if args.save_track_crops and (tr.frames - 1) % args.save_track_crops == 0:
                    d = os.path.join(args.out_dir, "track_crops", str(tr.id))
                    os.makedirs(d, exist_ok=True)
                    crop.save(os.path.join(d, f"{idx:06d}.png"))
            active = [t for t in active if idx - t.last <= args.max_age]
            shown = [(t, t.box) for t in active if t.last == idx]
        for t, box in shown:
            text, good = t.consensus()
            colour = (0, 255, 0) if t.reliable() else (0, 165, 255)   # green = enough frames agree
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"#{t.id} {text}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()

    print(f"{idx + 1} frames ({n_frames} reported), {len(tracks)} tracks\n")
    print(f"{'id':>3} {'frames':>6}  {'consensus':<14} plate-like  support  reliable  top raw readings")
    rows = []
    for t in sorted(tracks, key=lambda t: -t.frames):
        if t.frames < args.min_frames:
            continue
        text, good = t.consensus()
        top = ", ".join(f"{s} x{c}" for s, c in t.top_readings())
        sup, rel = t.support(), t.reliable()
        print(f"{t.id:>3} {t.frames:>6}  {text:<14} {'yes' if good else 'no ':<10} {sup:>7}  {'YES' if rel else 'no ':<8}  {top}")
        t.best_crop.save(os.path.join(args.out_dir, f"track_{t.id}_best.png"))
        rows.append((t.id, t.frames, text, int(good), top, sup, int(rel)))
    with open(os.path.join(args.out_dir, "tracks.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "frames", "consensus", "plate_like", "top_readings", "support", "reliable"])
        wr.writerows(rows)
    # The tracker often splits one plate into several tracks (occlusion, jitter). Merge plate-like
    # tracks whose readings differ by at most one character into one line per distinct plate.
    groups = []
    for t in sorted(tracks, key=lambda t: -t.frames):
        text, good = t.consensus()
        if t.frames < args.min_frames or not t.reliable():
            continue
        for g in groups:
            if levenshtein(g["text"], text) <= 1:
                g["frames"] += t.frames
                g["support"] += t.support()
                g["ids"].append(t.id)
                break
        else:
            groups.append({"text": text, "frames": t.frames, "support": t.support(), "ids": [t.id]})
    n_unrel = sum(1 for t in tracks if t.frames >= args.min_frames and not t.reliable())
    print(f"\ndistinct RELIABLE plates (>= {MIN_SUPPORT} frames agree and >= {MIN_SUPPORT_SHARE:.0%} of the track; "
          f"tracks merged when readings differ by <= 1 character; {n_unrel} other tracks ignored as unreliable):")
    for g in sorted(groups, key=lambda g: -g["support"]):
        print(f"  {g['text']:<14} {g['support']:>4} frames agree   (seen in {g['frames']} frames, tracks {g['ids']})")

    if args.dump_readings:
        with open(os.path.join(args.out_dir, "readings.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["track", "frame", "crop_w", "crop_h", "text"])
            for t in tracks:
                for fr, cw_, ch_, text in t.log:
                    wr.writerow([t.id, fr, cw_, ch_, text])
    print(f"\nwrote {args.out_dir}/annotated.mp4, tracks.csv and track_<id>_best.png")


if __name__ == "__main__":
    main()
