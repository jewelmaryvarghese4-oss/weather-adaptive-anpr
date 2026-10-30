"""
Option 4, step 1 (SEPARATE from every earlier result): follow each plate through the Ponmudi videos and save EVERY crop of it.
Same detector, same frame-to-frame linking as foggy_ft/build_video_tracks.py (which only kept crops of 6 tracks); nothing there is changed.

Writes datasets/fusion/tracks/<video>_<track>/f<frame>_w<width>.png   and   datasets/fusion/tracks_index.csv
    python foggy_ft/fusion/fusion_tracks.py          (run in WSL from uvm/)
"""
import argparse
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UVM = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(UVM)
D = os.path.join(ROOT, "datasets")
EXPORT = os.path.join(ROOT, "exports", "_stage", "lprnet_export")

import cv2
import numpy as np
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--videos", nargs="*", default=sorted(glob.glob(os.path.join(D, "gdrive_extra_photos", "*.mp4")) + glob.glob(os.path.join(D, "foggy_photos", "*.mp4"))))
ap.add_argument("--out", default=os.path.join(D, "fusion"))
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--min_w", type=int, default=36)
ap.add_argument("--min_obs", type=int, default=3, help="only tracks seen in at least this many frames are saved")
ap.add_argument("--limit_videos", type=int, default=0)
ap.add_argument("--max_frames", type=int, default=0, help="per video (smoke test)")
ap.add_argument("--device", default="")
a = ap.parse_args()


def iou(p, q):
    x1, y1, x2, y2 = max(p[0], q[0]), max(p[1], q[1]), min(p[2], q[2]), min(p[3], q[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)


def main():
    from ultralytics import YOLO
    det = YOLO(os.path.join(EXPORT, "detector", "yolo_plate_detector_15k_best.pt"))
    os.makedirs(os.path.join(a.out, "tracks"), exist_ok=True)
    vids = a.videos[:a.limit_videos] if a.limit_videos else a.videos
    index = []
    for vp in vids:
        vname = os.path.basename(vp)[9:15]
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        active, tracks, fi = [], [], -1
        bar = tqdm(total=min(total, a.max_frames) if a.max_frames else total, desc="video " + vname, unit="frame", dynamic_ncols=True)
        while True:
            ok, fr = cap.read()
            if not ok or (a.max_frames and fi + 1 >= a.max_frames):
                break
            fi += 1
            bar.update(1)
            if fi % a.stride:
                continue
            r = det.predict(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), conf=0.2, imgsz=1280, verbose=False, **({"device": a.device} if a.device else {}))[0]
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
                x1, y1, x2, y2 = (int(v) for v in b)
                crop = fr[max(0, y1):y2, max(0, x1):x2].copy()           # BGR, native resolution
                ob = (fi, b, crop)
                if best is None:
                    t = dict(obs=[ob])
                    active.append(t)
                    tracks.append(t)
                    used.add(len(active) - 1)
                else:
                    active[best]["obs"].append(ob)
                    used.add(best)
            active = [t for t in active if fi - t["obs"][-1][0] <= 3 * a.stride]
        bar.close()
        cap.release()
        kept = 0
        for ti, t in enumerate(tracks):
            if len(t["obs"]) < a.min_obs:
                continue
            kept += 1
            d = os.path.join(a.out, "tracks", "%s_%03d" % (vname, ti))
            os.makedirs(d, exist_ok=True)
            for f, b, c in t["obs"]:
                cv2.imwrite(os.path.join(d, "f%05d_w%d.png" % (f, int(b[2] - b[0]))), c)
            ws = [int(o[1][2] - o[1][0]) for o in t["obs"]]
            index.append([vname, "%s_%03d" % (vname, ti), len(t["obs"]), t["obs"][0][0], t["obs"][-1][0], max(ws)])
        print("video %s: %d frames, %d tracks, %d with >= %d crops" % (vname, fi + 1, len(tracks), kept, a.min_obs), flush=True)
    with open(os.path.join(a.out, "tracks_index.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "track_dir", "n_crops", "first_frame", "last_frame", "max_width_px"])
        w.writerows(index)
    print("FUSION_TRACKS_DONE", len(index), "tracks ->", a.out, flush=True)


if __name__ == "__main__":
    main()
