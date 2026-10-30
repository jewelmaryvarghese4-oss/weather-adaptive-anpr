"""
Cuts plate crops out of the session videos and groups them into tracks (one track = one car's plate across frames).

    python foggy_ft/build_video_tracks.py        (run in WSL from uvm/)

For every video: every 2nd frame -> YOLO plate detector (1280) -> boxes linked frame to frame by overlap -> every crop is read by the installed
reader in two views (plain + smoothed). A track's proposed label is the string that most readings agree on (valid Indian plate format only);
it is shown with its confidence share and MUST be confirmed by a person before training (check sheets are produced). Tracks whose proposed
label is within 1 edit of a verified foggy test plate are marked with that plate (so training can keep test plates out of their own fold).

Writes datasets/video_tracks/{crops/, tracks.csv, sheet_NN.png}
"""
import argparse
import csv
import glob
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
UVM = os.path.dirname(HERE)
ROOT = os.path.dirname(UVM)
D = os.path.join(ROOT, "datasets")
EXPORT = os.path.join(ROOT, "exports", "_stage", "lprnet_export")
sys.path.insert(0, os.path.join(EXPORT, "src"))
sys.path.insert(0, EXPORT)
os.environ.setdefault("TWO_LINE_MODE", "fixed")

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont

from plate_format import FORMATS_BY_LENGTH
from predict_plate import load_net, read_plate, smooth_view

ap = argparse.ArgumentParser()
ap.add_argument("--videos", nargs="*", default=sorted(glob.glob(os.path.join(D, "gdrive_extra_photos", "*.mp4")) + glob.glob(os.path.join(D, "foggy_photos", "*.mp4"))))
ap.add_argument("--out", default=os.path.join(D, "video_tracks"))
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--min_w", type=int, default=36)
ap.add_argument("--keep", type=int, default=10, help="crops kept per track")
a = ap.parse_args()


def conforms(t):
    return any(len(f) == len(t) and all((c.isalpha() if k == "L" else c.isdigit()) for c, k in zip(t, f)) for f in FORMATS_BY_LENGTH.get(len(t), []))


def lev(x, y):
    p = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        q = [i]
        for j, cy in enumerate(y, 1):
            q.append(min(p[j] + 1, q[j - 1] + 1, p[j - 1] + (cx != cy)))
        p = q
    return p[-1]


def iou(p, q):
    x1, y1, x2, y2 = max(p[0], q[0]), max(p[1], q[1]), min(p[2], q[2]), min(p[3], q[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)


os.makedirs(os.path.join(a.out, "crops"), exist_ok=True)
for f in glob.glob(os.path.join(a.out, "crops", "*")) + glob.glob(os.path.join(a.out, "sheet_*.png")):
    os.remove(f)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = load_net(os.path.join(EXPORT, "weights", "lprnet_avg_INSTALLED.pt"), dev)
from ultralytics import YOLO
det = YOLO(os.path.join(EXPORT, "detector", "yolo_plate_detector_15k_best.pt"))
ver = sorted({r["label"] for r in csv.DictReader(open(os.path.join(D, "foggy_ft", "foggy_labels_verified.csv"), encoding="utf-8")) if r["status"] in ("ok", "fix", "likely") and r["label"]})

tracks = []                      # each: dict(video, obs=[(frame, box, crop, [readings])])
for vp in a.videos:
    cap = cv2.VideoCapture(vp)
    vname = os.path.basename(vp)[9:15]
    active = []
    fi = -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fi += 1
        if fi % a.stride:
            continue
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        r = det.predict(rgb, conf=0.2, imgsz=1280, verbose=False)[0]
        used = set()
        for b in r.boxes.xyxy.tolist():
            if b[2] - b[0] < a.min_w:
                continue
            best, bi = None, 0.2
            for ti, t in enumerate(active):
                if ti in used or fi - t["obs"][-1][0] > 3 * a.stride:
                    continue
                s = iou(t["obs"][-1][1], b)
                if s > bi:
                    best, bi = ti, s
            crop = Image.fromarray(rgb).crop(tuple(int(v) for v in b))
            rd = [read_plate(net, crop, dev), read_plate(net, smooth_view(crop), dev)]
            ob = (fi, b, crop, [x[1] for x in rd], min(x[2] for x in rd))
            if best is None:
                t = dict(video=vname, obs=[ob])
                active.append(t)
                tracks.append(t)
                used.add(len(active) - 1)
            else:
                active[best]["obs"].append(ob)
                used.add(best)
        active = [t for t in active if fi - t["obs"][-1][0] <= 3 * a.stride]
    cap.release()
    print("video %s: %d frames read, tracks so far %d" % (vname, fi + 1, len(tracks)), flush=True)

rows, kept_tracks = [], []
for ti, t in enumerate(tracks):
    obs = t["obs"]
    valid = [s for o in obs for s in o[3] if conforms(s)]
    cons, share, n_agree = "", 0.0, 0
    if valid:
        cons, n_agree = Counter(valid).most_common(1)[0]
        share = n_agree / (2 * len(obs))
    widths = [o[1][2] - o[1][0] for o in obs]
    matched = ""
    if cons:
        d = sorted((lev(cons, v), v) for v in ver)
        if d[0][0] <= 1 and d[1][0] > 1:
            matched = d[0][1]
    ok = bool(cons) and n_agree >= 4 and share >= 0.3 and len(obs) >= 3
    rows.append([ti, t["video"], len(obs), obs[0][0], obs[-1][0], int(max(widths)), cons, round(share, 2), n_agree, matched, "proposed" if ok else "weak"])
    if ok:
        t["id"], t["cons"], t["share"], t["matched"] = ti, cons, share, matched
        kept_tracks.append(t)
with open(os.path.join(a.out, "tracks.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["track", "video", "frames_seen", "first_frame", "last_frame", "max_width_px", "proposed_label", "agree_share", "agree_count", "matches_verified_foggy_plate", "status"])
    w.writerows(rows)
print("tracks: %d | with a proposed label (>=4 agreeing readings, >=30%% share, >=3 frames): %d | of those matching a verified foggy plate: %d" % (
    len(tracks), len(kept_tracks), sum(1 for t in kept_tracks if t["matched"])), flush=True)

# crops: up to --keep per proposed track, spread over sizes (largest first, then every other one)
crop_rows = []
for t in kept_tracks:
    obs = sorted(t["obs"], key=lambda o: -(o[1][2] - o[1][0]))
    pick = obs[:a.keep // 2] + obs[a.keep // 2::2][:a.keep - a.keep // 2]
    t["files"] = []
    for k, o in enumerate(pick):
        name = "t%03d_%02d.png" % (t["id"], k)
        o[2].save(os.path.join(a.out, "crops", name))
        t["files"].append((name, o[2]))
        crop_rows.append([name, t["id"], t["cons"], t["matched"], int(o[1][2] - o[1][0])])
with open(os.path.join(a.out, "crops.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "track", "proposed_label", "matches_verified_foggy_plate", "width_px"])
    w.writerows(crop_rows)
print("crops saved:", len(crop_rows), flush=True)

# check sheets: one row per track = label + 4 crops (the largest and three others)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 26)
sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
RH, per = 130, 8
kept_tracks.sort(key=lambda t: (t["matched"] != "", t["video"], t["id"]))
for s in range(0, len(kept_tracks), per):
    chunk = kept_tracks[s:s + per]
    sheet = Image.new("RGB", (1500, RH * len(chunk)), "white")
    d = ImageDraw.Draw(sheet)
    for j, t in enumerate(chunk):
        y = j * RH
        d.text((6, y + 6), "T%d" % t["id"], fill=(200, 0, 0), font=font)
        d.text((6, y + 42), t["cons"], fill=(0, 90, 0), font=font)
        d.text((6, y + 80), "%s | %d frames | %d%% agree%s" % (t["video"], len(t["obs"]), int(100 * t["share"]),
                                                                " | = verified " + t["matched"] if t["matched"] else ""), fill=(110, 110, 110), font=sm)
        x = 330
        for name, im in (t["files"][:1] + t["files"][len(t["files"]) // 2:][:3]):
            c = im.copy()
            c.thumbnail((270, 90))
            sheet.paste(c, (x, y + 12))
            x += 285
        d.line([0, y + RH - 1, 1500, y + RH - 1], fill=(210, 210, 210))
    sheet.save(os.path.join(a.out, "sheet_%02d.png" % (s // per + 1)))
print("check sheets:", (len(kept_tracks) + per - 1) // per, flush=True)
