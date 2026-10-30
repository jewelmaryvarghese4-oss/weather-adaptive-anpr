"""
Video pipeline (two passes per video):  UVM (GoPro-trained)  ->  physics-guided dehaze, strong + exposure  ->  YOLOv11 plate detector  ->  LPRNet reader.
 Pass 1 (every frame):
  1. UVM GoPro model on the frame = the fog measurement.
  2. physics_dehaze.dehaze(gamma 1.25, smooth 0.03) inverts the fog law on the real frame, with the veil colour A smoothed over time (EMA)
     and a per-frame brightness gamma (target mean 0.46, same rule as exposure_match.py) also smoothed, so the video does not flicker.
  3. YOLOv11 plate detector (imgsz 1280, conf 0.25) on the dehazed frame; LPRNet (installed reader, 'select' view, format-corrected) on every plate;
     detections that do not read like a plate are dropped; boxes are linked from frame to frame (IoU) into raw tracks.
 Between the passes:
  4. raw tracks of the same car that were cut by a short drop-out (<= --merge_gap frames, nearby box) are joined into one "vehicle";
     each vehicle's FINAL reading is the most voted reading over all its frames.
 Pass 2 (videos):
  5. the videos are drawn with the final reading of each vehicle from its first frame on (it never changes while the video plays): box label and a
     top-left inset with the enlarged plate crop of the current frame, the final number and its votes; slow motion (--slow) while a plate is detected.
Writes into --out:  <video>_dehazed_annotated.mp4, <video>_original_vs_dehazed.mp4, <video>_detections.csv, plate_tracks.csv, crops/, summary.txt
    python real_scratch/run_video_pipeline.py --input /mnt/c/.../finalin/foggyvideo --out /mnt/c/.../finalout/fogvidphy      (WSL, from uvm/)
"""
import argparse
import csv
import glob
import os
import sys
from collections import Counter

UVM = "/mnt/c/projects/jewwww/uvm"
EXPORT = "/mnt/c/projects/jewwww/exports/_stage/lprnet_export"
sys.path.insert(0, UVM)
sys.path.insert(0, os.path.join(UVM, "foggy_ft", "clean"))
sys.path.insert(0, os.path.join(EXPORT, "src"))
sys.path.insert(0, EXPORT)
os.environ.setdefault("TWO_LINE_MODE", "fixed")

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import test_uvmnet as uvm
from model_uvmnet import UVMNet
from physics_dehaze import dehaze
from predict_plate import load_net, looks_like_plate, read_plate_view

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--checkpoint", default=os.path.join(UVM, "checkpoints_ponmudi_gopro", "uvmnet_combined_final.pt"))
ap.add_argument("--gamma", type=float, default=1.25)
ap.add_argument("--smooth", type=float, default=0.03)
ap.add_argument("--target_brightness", type=float, default=0.46)
ap.add_argument("--ema_A", type=float, default=0.2)
ap.add_argument("--ema_gamma", type=float, default=0.15)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--max_frames", type=int, default=0, help="per video (smoke test)")
ap.add_argument("--slow", type=int, default=4, help="while a plate is detected every frame is shown this many times (slow motion); 1 = real speed")
ap.add_argument("--slow_hold", type=int, default=4, help="keep the slow motion this many frames after the last detection (smooth ending)")
ap.add_argument("--inset_max", type=int, default=3, help="most plates shown in the top-left inset at once")
ap.add_argument("--merge_gap", type=int, default=15, help="join raw tracks of the same car cut by a drop-out of at most this many frames")
a = ap.parse_args()


def iou(p, q):
    x1, y1, x2, y2 = max(p[0], q[0]), max(p[1], q[1]), min(p[2], q[2]), min(p[3], q[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)


def center_dist(p, q):
    return float(np.hypot((p[0] + p[2]) / 2 - (q[0] + q[2]) / 2, (p[1] + p[3]) / 2 - (q[1] + q[3]) / 2))


def writer(path, fps, size):
    for fourcc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w
    raise RuntimeError("cannot open a video writer for " + path)


def text_outline(img, txt, org, scale, color, thick=2):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def add_inset(out, items, max_items):
    """Top-left inset: for each detected plate the enlarged crop of this frame, the FINAL number and its votes. items = [(PIL crop, number, votes text)]."""
    H, W = out.shape[:2]
    y = 12
    for crop, txt, sub in items[:max_items]:
        ch = max(54, H // 8)
        cw = min(int(crop.width * ch / max(crop.height, 1)), W // 3)
        c = cv2.resize(cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR), (cw, ch), interpolation=cv2.INTER_CUBIC)
        fs = max(0.8, H / 560)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)
        pw, ph = cw + tw + 36, ch + 16
        roi = out[y:y + ph, 10:10 + pw]
        if roi.shape[0] < ph or roi.shape[1] < pw:
            break
        out[y:y + ph, 10:10 + pw] = (roi * 0.25).astype(np.uint8)                    # dark, mostly opaque panel
        cv2.rectangle(out, (10, y), (10 + pw, y + ph), (0, 255, 0), 2)
        out[y + 8:y + 8 + ch, 18:18 + cw] = c
        tx = 18 + cw + 14
        text_outline(out, txt, (tx, y + 8 + ch // 2 - 2), fs, (0, 255, 0), 3)
        cv2.putText(out, sub, (tx, y + 8 + ch // 2 + int(26 * fs)), cv2.FONT_HERSHEY_SIMPLEX, max(0.45, fs * 0.5), (200, 200, 200), 1, cv2.LINE_AA)
        y += ph + 8


def draw_boxes(img, dets):
    out = img.copy()
    lw = max(2, img.shape[1] // 400)
    fs = max(0.5, img.shape[1] / 1600)
    for b, dc, label in dets:
        b = [int(v) for v in b]
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), lw)
        label = "%s (%.2f)" % (label, dc)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        ty = max(th + 6, b[1] - 6)
        cv2.rectangle(out, (b[0], ty - th - 6), (b[0] + tw + 8, ty + 4), (0, 0, 0), -1)
        cv2.putText(out, label, (b[0] + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def merge_tracks(raw):
    """Joins raw tracks cut by a short drop-out (the same car, nearly the same place) -> vehicle id per raw track."""
    n = len(raw)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    has_succ = set()
    for j in sorted(range(n), key=lambda k: raw[k]["obs"][0]["fi"]):
        fj, bj = raw[j]["obs"][0]["fi"], raw[j]["obs"][0]["b"]
        best, bs = None, 1e9
        for i in range(n):
            if i == j or i in has_succ:
                continue
            li = raw[i]["obs"][-1]
            gap = fj - li["fi"]
            if gap < 1 or gap > a.merge_gap:
                continue
            wmax = max(bj[2] - bj[0], li["b"][2] - li["b"][0])
            d = center_dist(li["b"], bj)
            if d < 2.0 * wmax and gap + d / wmax < bs:
                best, bs = i, gap + d / wmax
        if best is not None:
            has_succ.add(best)
            parent[find(j)] = find(best)
    roots = sorted({find(i) for i in range(n)}, key=lambda r: min(raw[k]["obs"][0]["fi"] for k in range(n) if find(k) == r))
    ids = {r: k for k, r in enumerate(roots)}
    return [ids[find(i)] for i in range(n)]


def main():
    from ultralytics import YOLO
    dev = torch.device("cuda")
    torch.set_grad_enabled(False)
    net = UVMNet(n_channels=3, backend="cuda", uvmb_size=32).to(dev)
    net.load_state_dict(torch.load(a.checkpoint, map_location=dev))
    net.eval()
    reader = load_net(os.path.join(EXPORT, "weights", "lprnet_avg_INSTALLED.pt"), dev)
    det = YOLO(os.path.join(EXPORT, "detector", "yolo_plate_detector_15k_best.pt"))
    os.makedirs(os.path.join(a.out, "crops"), exist_ok=True)
    vids = sorted(glob.glob(os.path.join(a.input, "*.mp4")) + glob.glob(os.path.join(a.input, "*.mov")) + glob.glob(os.path.join(a.input, "*.avi")))
    track_rows, summary = [], []
    for vp in vids:
        vname = os.path.splitext(os.path.basename(vp))[0]
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if a.max_frames:
            total = min(total, a.max_frames)

        # ------------------------------------------------------------------ pass 1: de-fog, detect, read, link
        frames, raw, active = [], [], []
        A_s, g_s, fi = None, None, -1
        bar = tqdm(total=total, desc=vname + " (read)", unit="frame", dynamic_ncols=True)
        while True:
            ok, fr = cap.read()
            if not ok or (a.max_frames and fi + 1 >= a.max_frames):
                break
            fi += 1
            bar.update(1)
            rgb = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            rawimg = cv2.cvtColor(np.asarray(uvm.process_image(net, rgb, dev)), cv2.COLOR_RGB2BGR)
            torch.cuda.empty_cache()
            out, A_est = dehaze(fr, rawimg, gamma=a.gamma, smooth=a.smooth, A_fixed=A_s)
            A_s = A_est if A_s is None else (1 - a.ema_A) * A_s + a.ema_A * A_est
            lum = float(np.clip(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32).mean() / 255, 0.02, 0.98))
            g = float(np.clip(np.log(a.target_brightness) / np.log(lum), 0.6, 1.0))
            g_s = g if g_s is None else (1 - a.ema_gamma) * g_s + a.ema_gamma * g
            out = np.clip((out.astype(np.float32) / 255.0) ** g_s * 255 + 0.5, 0, 255).astype(np.uint8)
            img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            r = det.predict(img, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
            dets, used = [], set()
            for b, dc in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                crop = img.crop(tuple(int(v) for v in b))
                rawtxt, fmt, rc, view = read_plate_view(reader, crop, dev, "select")
                if not looks_like_plate(fmt):
                    continue
                best, bi = None, 0.3
                for ti, t in enumerate(active):
                    if ti in used or fi - t["obs"][-1]["fi"] > 5:
                        continue
                    s = iou(t["obs"][-1]["b"], b)
                    if s > bi:
                        best, bi = ti, s
                ob = {"fi": fi, "b": b, "dc": dc, "fmt": fmt, "raw": rawtxt, "rc": rc, "view": view, "crop": crop}
                if best is None:
                    t = {"obs": [ob]}
                    active.append(t)
                    raw.append(t)
                    used.add(len(active) - 1)
                    ob["rt"] = len(raw) - 1
                else:
                    active[best]["obs"].append(ob)
                    used.add(best)
                    ob["rt"] = raw.index(active[best])
                dets.append(ob)
            active = [t for t in active if fi - t["obs"][-1]["fi"] <= 5]
            frames.append({"fr": fr, "out": out, "dets": dets})
        bar.close()
        cap.release()

        # ------------------------------------------------------------------ vehicles and their final (most voted) reading
        veh_of_raw = merge_tracks(raw) if raw else []
        veh = {}
        for rt, t in enumerate(raw):
            veh.setdefault(veh_of_raw[rt], []).extend(t["obs"])
        final = {}
        for v, obs in veh.items():
            votes = Counter(o["fmt"] for o in obs)
            top = max(votes.values())
            cands = [s for s, c in votes.items() if c == top]
            text = max(cands, key=lambda s: sum(o["dc"] for o in obs if o["fmt"] == s))          # ties: the reading the detector was surest about
            final[v] = {"text": text, "votes": votes[text], "n": len(obs), "obs": obs, "frags": sum(1 for x in veh_of_raw if x == v), "top": votes.most_common(3)}

        # ------------------------------------------------------------------ pass 2: draw the videos with the final readings
        w_ann = writer(os.path.join(a.out, vname + "_dehazed_annotated.mp4"), fps, (W, H))
        w_cmp = writer(os.path.join(a.out, vname + "_original_vs_dehazed.mp4"), fps, (2 * W, H))
        rows, slow_until = [], -1
        for fi, f in enumerate(frames):
            fr, out, dets = f["fr"], f["out"], f["dets"]
            for ob in dets:
                v = veh_of_raw[ob["rt"]]
                rows.append([fi, "%.2f" % (fi / fps), v, ",".join("%d" % x for x in ob["b"]), "%.3f" % ob["dc"], ob["fmt"], ob["raw"], "%.3f" % ob["rc"], ob["view"], final[v]["text"]])
            ann = draw_boxes(out, [(o["b"], o["dc"], final[veh_of_raw[o["rt"]]]["text"]) for o in dets])
            if dets:
                items = [(o["crop"], final[veh_of_raw[o["rt"]]]["text"], "most voted: %d of %d frames" % (final[veh_of_raw[o["rt"]]]["votes"], final[veh_of_raw[o["rt"]]]["n"]))
                         for o in sorted(dets, key=lambda o: -o["dc"])]
                add_inset(ann, items, a.inset_max)
                slow_until = fi + a.slow_hold
            slow = a.slow if (dets or fi <= slow_until) else 1
            text_outline(ann, "%.1f s" % (fi / fps) + ("   SLOW MOTION x%d" % slow if slow > 1 else ""), (12, H - 14), max(0.6, H / 900), (255, 255, 255), 2)
            left = fr.copy()
            text_outline(left, "ORIGINAL", (12, 34), 1.0, (255, 255, 255), 2)
            text_outline(left, "%.1f s" % (fi / fps), (12, H - 14), max(0.6, H / 900), (255, 255, 255), 2)
            cmp_frame = np.hstack([left, ann])
            for _ in range(slow):                 # slow motion while a plate is on screen: the same frame is shown several times
                w_ann.write(ann)
                w_cmp.write(cmp_frame)
        w_ann.release(); w_cmp.release()
        with open(os.path.join(a.out, vname + "_detections.csv"), "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["frame", "time_s", "vehicle", "box_x1,y1,x2,y2", "detector_confidence", "reading_this_frame", "reading_plain", "reader_confidence", "view_used", "final_reading_of_vehicle"])
            wr.writerows(rows)
        for v in sorted(final):
            d = final[v]
            obs = d["obs"]
            best = max(obs, key=lambda o: o["dc"])
            best["crop"].save(os.path.join(a.out, "crops", "%s_vehicle%02d.png" % (vname, v)))
            track_rows.append([vname, v, len(obs), min(o["fi"] for o in obs), max(o["fi"] for o in obs), d["text"], d["votes"], "%.2f" % (d["votes"] / len(obs)),
                               "%.2f" % (sum(o["dc"] for o in obs) / len(obs)), d["frags"], "; ".join("%s x%d" % (s, c) for s, c in d["top"])])
        summary.append("%s: %d frames, %d plate detections kept, %d raw tracks -> %d vehicles" % (vname, len(frames), len(rows), len(raw), len(final)))
        print(summary[-1], flush=True)
        del frames
    with open(os.path.join(a.out, "plate_tracks.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video", "vehicle", "frames_seen", "first_frame", "last_frame", "final_reading_most_voted", "votes_for_it", "vote_share", "mean_detector_conf", "raw_tracks_joined", "top_readings"])
        wr.writerows(track_rows)
    txt = "\n".join(summary) + "\n\npipeline: UVM (GoPro-trained) -> physics-guided dehaze (strong gamma %.2f, smooth %.2f, veil colour and brightness smoothed over time) -> YOLOv11 plate detector (imgsz %d, conf %.2f) -> LPRNet (installed reader, select view, format-corrected)\n" % (a.gamma, a.smooth, a.imgsz, a.conf)
    txt += ("videos: slow motion x%d while a plate is detected (+%d frames afterwards); top-left inset (up to %d plates) with the enlarged crop of the current frame, the FINAL number "
            "(the most voted reading of that vehicle over the whole video, fixed from its first frame) and its votes; raw tracks cut by a drop-out of <= %d frames are joined into one vehicle\n") % (a.slow, a.slow_hold, a.inset_max, a.merge_gap)
    open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    print("VIDEO_PIPELINE_DONE")


if __name__ == "__main__":
    main()
