"""
Browser interface for the plate reader: pick a video or photos from your folders (no commands to type),
watch the plates being read live, and look through the results of earlier runs.

Start it once from WSL2 and leave it running:
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python plate_app.py
then open http://localhost:8000 in your Windows browser. Ctrl+C in the terminal stops it.

Videos go through the same detector -> tracker -> reader -> vote as live_plates.py (LiveReader), photos through
the same detector + read_plate() as read_plate.py (no dehazing). Every run is saved under app_runs/<run>/:
plates.csv, tracks.csv, the clearest crop and a snapshot of every confirmed plate, crops of tracked vehicles
whose plate was never confirmed, and (optionally) the annotated recording.

Only the Python standard library is used for the web server, nothing extra to install.
"""
import argparse
import collections
import csv
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from PIL import Image, ImageOps

from async_writer import AsyncWriter
from live_plates import LiveReader, build_parser, load_models, open_source
from read_plate import looks_like_plate, read_plate

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "app_runs")
# uploads go to the Linux disk: much faster to write and to read back than /mnt/c
UPLOAD_DIR = os.path.expanduser("~/plate_app_uploads")
KEEP_UPLOADS = 5             # older uploads are deleted (their results stay in app_runs)
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm", ".mpg", ".mpeg", ".3gp", ".ts"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIEW_WIDTH = 1280            # width of the live view sent to the browser
UNCONFIRMED_MIN_FRAMES = 20  # same as live_plates.py --min_unconfirmed_frames


def make_preview(src, out, stop):
    """Writes an H.264, 720p copy of the video that any browser can play (the phone videos are HEVC, which many browsers
    cannot). Takes about 8 s for a clip of a minute. Returns True when the file is ready."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False
    tmp = out[:-4] + ".part.mp4"
    cmd = [exe, "-y", "-loglevel", "error", "-i", src, "-vf", "scale=1280:-2", "-an", "-c:v", "libx264", "-preset", "ultrafast",
           "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while proc.poll() is None:
        if stop.is_set():
            proc.kill()
            return False
        time.sleep(0.2)
    if proc.returncode != 0 or not os.path.exists(tmp):
        return False
    os.replace(tmp, out)
    return True


def safe_name(name):
    name = os.path.basename(name.replace("\\", "/"))
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "file"


def to_wsl_path(p):
    """C:\\projects\\x.mp4 -> /mnt/c/projects/x.mp4 (paths pasted from Windows Explorer)."""
    p = p.strip().strip('"')
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        return f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")
    return p


def display_name(path):
    """File name without the upload prefix (20261008_101500_1_)."""
    return re.sub(r"^\d{8}_\d{6}_\d+_", "", os.path.basename(path))


def snapshot(meta):
    """JSON copy of a run's meta that the worker thread may be changing at this moment."""
    for _ in range(20):
        try:
            return json.loads(json.dumps(meta))
        except RuntimeError:          # dict changed size during the copy: try again
            time.sleep(0.005)
    return {}


def sniff_kind(path):
    """'video' / 'photo' / None from the first bytes of a file (for files saved without an extension, e.g. '2')."""
    try:
        with open(path, "rb") as f:
            h = f.read(16)
    except OSError:
        return None
    if (h[4:8] == b"ftyp"                                        # MP4 / MOV / M4V / 3GP
            or h[:4] == bytes.fromhex("1a45dfa3")                 # MKV / WebM
            or (h[:4] == b"RIFF" and h[8:12] == b"AVI ")          # AVI
            or h[:4] == bytes.fromhex("000001ba")                 # MPEG program stream
            or h[:4] == bytes.fromhex("3026b275")                 # WMV
            or h[:4] == b"OggS" or h[:3] == b"FLV"):
        return "video"
    if (h[:3] == bytes.fromhex("ffd8ff")                          # JPEG
            or h[:8] == bytes.fromhex("89504e470d0a1a0a")         # PNG
            or h[:2] == b"BM"                                     # BMP
            or (h[:4] == b"RIFF" and h[8:12] == b"WEBP")          # WebP
            or h[:4] in (bytes.fromhex("49492a00"), bytes.fromhex("4d4d002a"))):   # TIFF
        return "photo"
    return None


def kind_of(path):
    """video / photo by extension; a file with no extension is recognised by its content."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
        return "photo"
    if ext in VIDEO_EXT:
        return "video"
    if ext == "" and os.path.isfile(path):
        return sniff_kind(path)
    return None


def natural_key(name):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", name)]


def write_jpg(path, bgr, quality=88):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp, path)          # a browser never sees a half-written file


def save_png(img, path):
    """Saves a PIL image atomically (write to a temp file, then rename)."""
    tmp = path + ".tmp"
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


class Run:
    def __init__(self, kind, sources, settings):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        names = [display_name(s) for s in sources]
        base = os.path.splitext(names[0])[0] if len(sources) == 1 else f"{len(sources)}_photos"
        rid = f"{stamp}_{safe_name(base)}"
        k = 2
        while os.path.exists(os.path.join(RUNS_DIR, rid)):
            rid = f"{stamp}_{safe_name(base)}_{k}"
            k += 1
        self.id, self.dir = rid, os.path.join(RUNS_DIR, rid)
        os.makedirs(self.dir)
        self.stop = threading.Event()
        self.meta = {"id": rid, "kind": kind, "name": base if len(sources) == 1 else f"{len(sources)} photos",
                     "sources": names, "settings": settings,
                     "status": "queued", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "events": [], "unconfirmed": [], "photos": [], "progress": 0.0, "error": None}
        self.sources = sources

    def save(self):
        tmp = os.path.join(self.dir, "run.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=1)
        os.replace(tmp, os.path.join(self.dir, "run.json"))


class Engine:
    """Owns the models; runs one job at a time on a worker thread (the GPU is shared)."""

    def __init__(self, lprnet_ckpt, yolo_ckpt):
        self.lprnet_ckpt, self.yolo_ckpt = lprnet_ckpt, yolo_ckpt
        self.lprnet, self.yolo, self.dev = load_models(lprnet_ckpt, yolo_ckpt)
        self.pool = None                 # parallel_reader.WorkerPool, created the first time a run asks for several workers
        self.jobs = queue.Queue()
        self.queued = []                 # runs waiting (for the page)
        self.current = None
        self.lock = threading.Lock()
        self.frame_cond = threading.Condition()
        self.frame_jpg, self.frame_seq = None, 0
        self.viewers = 0
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- job queue ------------------------------------------------------------------------------
    def submit(self, run):
        with self.lock:
            self.queued.append(run)
        run.save()
        self.jobs.put(run)

    def stop_current(self):
        with self.lock:
            if self.current:
                self.current.stop.set()
            for r in self.queued:
                r.stop.set()

    def _worker(self):
        while True:
            run = self.jobs.get()
            with self.lock:
                self.queued = [r for r in self.queued if r is not run]
                self.current = run
            if run.stop.is_set():
                run.meta["status"] = "stopped"
            else:
                run.meta["status"] = "running"
                run.meta["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
                try:
                    (self._video if run.meta["kind"] == "video" else self._photos)(run)
                    run.meta["status"] = "stopped" if run.stop.is_set() else "done"
                except Exception as e:     # keep the server alive; show the error on the page
                    traceback.print_exc()
                    run.meta["status"], run.meta["error"] = "error", f"{type(e).__name__}: {e}"
            run.meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            run.save()
            with self.lock:
                self.current = None
            self._publish(None)
            prune_uploads()

    def _publish(self, bgr):
        jpg = None
        if bgr is not None:
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpg = buf.tobytes() if ok else None
        with self.frame_cond:
            self.frame_jpg, self.frame_seq = jpg, self.frame_seq + 1
            self.frame_cond.notify_all()

    def _get_pool(self, n):
        """The worker processes stay alive between runs (loading the models takes a while); rebuilt if the number changes."""
        from parallel_reader import WorkerPool
        if self.pool is not None and (self.pool.n != n or not self.pool.alive()):
            self.pool.close()
            self.pool = None
        if self.pool is None:
            self.pool = WorkerPool(n, self.lprnet_ckpt, self.yolo_ckpt)
        return self.pool

    # ---- video ----------------------------------------------------------------------------------
    def _video(self, run):
        st, src, meta = run.meta["settings"], run.sources[0], run.meta
        args = build_parser().parse_args(["--source", src, "--no_show"])
        try:
            cap, is_file = open_source(src)
        except SystemExit as e:
            raise RuntimeError(str(e)) from None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        imgsz = st.get("imgsz", "auto")
        args.yolo_imgsz = (1280 if max(W, H) >= 2500 else 640) if imgsz == "auto" else int(imgsz)
        realtime = st.get("mode", "live") == "live"
        meta["video"] = {"width": W, "height": H, "fps": round(fps, 2), "frames": total,
                         "duration_s": round(total / fps, 1) if total else None, "yolo_imgsz": args.yolo_imgsz}
        os.makedirs(os.path.join(run.dir, "crops"), exist_ok=True)
        os.makedirs(os.path.join(run.dir, "snaps"), exist_ok=True)

        vscale = min(1.0, VIEW_WIDTH / W)
        vsize = (int(W * vscale), int(H * vscale))
        writer = None
        if st.get("record", True):
            writer = AsyncWriter(os.path.join(run.dir, "annotated.mp4"), fps, vsize)
            meta["recording"] = "annotated.mp4"
        ring = collections.deque(maxlen=args.max_age + 3)    # recent undrawn frames, for plate snapshots
        by_track = {}                                        # track id -> its event dict

        def on_announce(tr, vt, old):
            if old is None:
                ev = {"t": round(vt, 1), "plate": tr.text, "track": tr.id, "corrected_from": None,
                      "crop": None, "snap": None, "reads": tr.reads, "support": tr.freq[tr.text]}
                frame = next((f for i, f in reversed(ring) if i == tr.last), ring[-1][1] if ring else None)
                if frame is not None:
                    snap = frame.copy()
                    x1, y1, x2, y2 = (int(v * vscale) for v in tr.box)
                    cv2.rectangle(snap, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 255, 0), 3)
                    ev["snap"] = f"snaps/track{tr.id}.jpg"
                    write_jpg(os.path.join(run.dir, ev["snap"]), snap)
                if tr.best_crop is not None:
                    ev["crop"] = f"crops/track{tr.id}.png"
                    save_png(tr.best_crop, os.path.join(run.dir, ev["crop"]))
                by_track[tr.id] = ev
                meta["events"].append(ev)
            elif tr.id in by_track:
                ev = by_track[tr.id]
                ev["corrected_from"], ev["plate"] = old, tr.text
                ev["support"] = tr.freq[tr.text]

        if st.get("compare", True) and realtime and is_file and not run.stop.is_set():
            meta["note"] = "Preparing the original video for the side-by-side view…"
            run.save()
            if make_preview(src, os.path.join(run.dir, "preview.mp4"), run.stop):
                meta["preview"] = "preview.mp4"
            meta.pop("note", None)
            run.save()
        reader = LiveReader(self.lprnet, self.yolo, self.dev, fps, args, keep_crops=True, on_announce=on_announce)
        workers = max(1, int(st.get("workers", 1)))
        meta["workers"] = workers
        stats = {"processed": 0, "skipped": 0}
        t_start = time.perf_counter()
        proc_times = collections.deque(maxlen=30)

        def remember(idx, frame):
            """Called right before a frame is tracked, so a plate announced on this frame can find it for its snapshot."""
            small = cv2.resize(frame, vsize, interpolation=cv2.INTER_AREA) if vscale < 1 else frame.copy()
            ring.append((idx, small))

        def single_frames():
            idx = -1
            while not run.stop.is_set():
                if is_file and realtime and stats["processed"]:
                    target = int((time.perf_counter() - t_start) * fps)
                    while idx + 1 < target:
                        if not cap.grab():
                            break
                        idx += 1
                        stats["skipped"] += 1
                ok, frame = cap.read()
                if not ok:
                    return
                idx += 1
                t0 = time.perf_counter()
                remember(idx, frame)
                reader.step(frame, idx)
                stats["processed"] += 1
                proc_times.append(time.perf_counter() - t0)
                yield idx, frame

        def parallel_frames():
            from parallel_reader import run_parallel
            meta["note"] = f"Starting {workers} workers (loading the models takes about 20 s the first time)…"
            run.save()
            pool = self._get_pool(workers)
            meta.pop("note", None)
            t_prev = time.perf_counter()
            for idx, frame, info in run_parallel(pool, cap, reader, args, fps, is_file=is_file, realtime=realtime,
                                                 max_lag=1.0, stop=run.stop, on_frame=remember):
                now = time.perf_counter()
                proc_times.append(now - t_prev)
                t_prev = now
                stats["processed"] += 1
                stats["skipped"] = info["skipped"]
                yield idx, frame

        last_save = time.perf_counter()
        idx = -1
        written = {"idx": -1, "frame": None}          # the last frame put into the recording
        for idx, frame in (parallel_frames() if workers > 1 else single_frames()):
            proc_fps = len(proc_times) / max(sum(proc_times), 1e-6)
            if writer or self.viewers:
                drawn = ring[-1][1].copy()
                draw_scaled(reader, drawn, idx, vscale)
                if writer:
                    # frames that were skipped get the previous picture again, so the recording keeps the video's real
                    # speed and length (otherwise 4 s of processed frames would play as a 4 s fast-forward of 30 s)
                    writer.write(written["frame"] if written["frame"] is not None else drawn, idx - written["idx"] - 1)
                    writer.write(drawn)
                    written["idx"], written["frame"] = idx, drawn
                if self.viewers:
                    self._publish(drawn)
            meta["progress"] = round((idx + 1) / total, 4) if total else 0
            meta["stats"] = {"video_time_s": round(idx / fps, 1), "processed": stats["processed"], "skipped": stats["skipped"],
                             "fps": round(proc_fps, 1), "wall_s": round(time.perf_counter() - t_start, 1)}
            if time.perf_counter() - last_save > 5:
                run.save()
                last_save = time.perf_counter()
        processed, skipped = stats["processed"], stats["skipped"]
        cap.release()
        reader.finish()
        with open(os.path.join(run.dir, "samples.json"), "w") as f:          # where each plate was seen (also useful for debugging)
            json.dump({str(t): [[i, [round(v, 1) for v in b]] for i, b in smp] for t, smp in reader.samples.items()}, f)
        if writer:
            writer.release()
            if stats["skipped"] > 0 and is_file:
                # frames were skipped while the run was live: make a smooth version from the original frames
                try:
                    from render_video import render_smooth
                    smooth = os.path.join(run.dir, "annotated_smooth.mp4")
                    meta["note"] = "Finishing the annotated video…"
                    meta["progress"] = 1.0
                    run.save()
                    end = idx if (run.stop.is_set() or not total) else max(idx, total - 1)     # a finished run: to the last frame
                    # the 720p copy decodes many times faster than the original (4K takes minutes), so the pass uses it
                    prev = os.path.join(run.dir, "preview.mp4")
                    if not os.path.exists(prev) and not run.stop.is_set():
                        make_preview(src, prev, run.stop)
                    render_smooth(prev if os.path.exists(prev) else src, smooth, reader, fps, vsize, end,
                                  progress=lambda f: meta.__setitem__("note", f"Finishing the annotated video… {int(100 * f)} %"),
                                  orig_width=W)
                    os.replace(smooth, os.path.join(run.dir, "annotated.mp4"))
                except Exception:                                  # keep the recording made during the run
                    traceback.print_exc()
                meta.pop("note", None)

        # final pass: the clearest crop of each confirmed plate may have improved after it was announced
        for ev in meta["events"]:
            tr = reader.all_tracks.get(ev["track"])
            if tr is not None and tr.best_crop is not None:
                ev["crop"] = f"crops/track{tr.id}.png"
                save_png(tr.best_crop, os.path.join(run.dir, ev["crop"]))
        announced_ids = {ev["track"] for ev in meta["events"]}
        os.makedirs(os.path.join(run.dir, "unconfirmed"), exist_ok=True)
        with open(os.path.join(run.dir, "tracks.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["track", "first_s", "last_s", "frames_seen", "reads", "max_width_px", "vote", "reliable",
                        "announced_as", "distinct_readings", "top_readings"])
            for tid, t in sorted(reader.all_tracks.items()):
                top = "; ".join(f"{s} x{c}" for s, c in t.freq.most_common(3))
                w.writerow([tid, f"{t.first_idx / fps:.1f}", f"{t.last / fps:.1f}", t.seen, t.reads, int(t.max_w),
                            t.text, int(t.reliable), t.announced or "", len(t.freq), top])
                if not t.announced and tid not in announced_ids and t.seen >= UNCONFIRMED_MIN_FRAMES \
                        and t.best_crop is not None:
                    crop = f"unconfirmed/track{tid}.png"
                    save_png(t.best_crop, os.path.join(run.dir, crop))
                    meta["unconfirmed"].append({"track": tid, "first_s": round(t.first_idx / fps, 1),
                                                "last_s": round(t.last / fps, 1), "frames": t.seen,
                                                "vote": t.text, "top": top, "crop": crop,
                                                "w": t.best_crop.width, "h": t.best_crop.height})
        with open(os.path.join(run.dir, "plates.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_time_s", "plate", "track", "corrected_from"])
            for ev in meta["events"]:
                w.writerow([f"{ev['t']:.1f}", ev["plate"], ev["track"], ev["corrected_from"] or ""])
        wall = time.perf_counter() - t_start
        meta["progress"] = 1.0 if not run.stop.is_set() else meta["progress"]
        meta["stats"] = {"video_time_s": round(idx / fps, 1), "processed": processed, "skipped": skipped,
                         "fps": round(processed / max(wall, 1e-6), 1), "wall_s": round(wall, 1),
                         "tracks": len(reader.all_tracks)}

    # ---- photos ---------------------------------------------------------------------------------
    def _photos(self, run):
        meta = run.meta
        conf_min = 0.25                                       # read_plate.py's default
        os.makedirs(os.path.join(run.dir, "crops"), exist_ok=True)
        os.makedirs(os.path.join(run.dir, "annotated"), exist_ok=True)
        for n, src in enumerate(run.sources):
            if run.stop.is_set():
                break
            name = display_name(src)
            stem = f"{n:03d}_{safe_name(os.path.splitext(name)[0])}"
            entry = {"name": name, "image": None, "plates": [], "whole_image": False}
            try:
                img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
            except Exception as e:
                entry["error"] = f"could not open: {e}"
                meta["photos"].append(entry)
                continue
            boxes = []
            for r in self.yolo.predict(img, conf=conf_min, verbose=False):
                boxes += [(tuple(b), c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
            if not boxes:      # no plate found: maybe the photo IS a plate crop
                boxes = [((0, 0, img.width, img.height), None)]
                entry["whole_image"] = True
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            th = max(2, round(max(img.size) / 600))
            for k, (box, conf) in enumerate(boxes):
                crop = img.crop(tuple(int(v) for v in box))
                _, text = read_plate(self.lprnet, crop, self.dev)
                plate_like = looks_like_plate(text)
                cp = f"crops/{stem}_{k}.png"
                save_png(crop, os.path.join(run.dir, cp))
                entry["plates"].append({"text": text, "conf": None if conf is None else round(conf, 2),
                                        "plate_like": plate_like, "crop": cp, "w": crop.width, "h": crop.height})
                if not entry["whole_image"]:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    col = (0, 220, 0) if plate_like else (0, 165, 255)
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), col, th)
                    cv2.putText(bgr, text or "?", (x1, max(20 * th, y1 - 4 * th)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5 * th, col, th, cv2.LINE_AA)
            entry["image"] = f"annotated/{stem}.jpg"
            scale = min(1.0, 1600 / max(bgr.shape[:2]))
            write_jpg(os.path.join(run.dir, entry["image"]),
                      cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr)
            meta["photos"].append(entry)
            meta["progress"] = round((n + 1) / len(run.sources), 4)
        with open(os.path.join(run.dir, "plates.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["photo", "plate", "looks_like_plate", "detector_conf", "crop_w", "crop_h"])
            for e in meta["photos"]:
                for p in e["plates"]:
                    w.writerow([e["name"], p["text"], int(p["plate_like"]), p["conf"] if p["conf"] is not None else "",
                                p["w"], p["h"]])


def draw_scaled(reader, img, idx, s):
    """LiveReader.draw() for a frame that was shrunk by s (boxes are in source coordinates)."""
    for t in reader.active:
        if t.last != idx:
            continue
        x1, y1, x2, y2 = (int(v * s) for v in t.box)
        col = (0, 220, 0) if t.reliable else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        if t.text:
            cv2.putText(img, t.text, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, t.text, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    cv2.putText(img, f"{idx / reader.fps:6.1f} s", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, f"{idx / reader.fps:6.1f} s", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                cv2.LINE_AA)


def prune_uploads():
    try:
        files = sorted((e for e in os.scandir(UPLOAD_DIR) if e.is_file()), key=lambda e: e.stat().st_mtime,
                       reverse=True)
    except FileNotFoundError:
        return
    busy = {os.path.abspath(s) for r in ENGINE.queued for s in r.sources}
    if ENGINE.current:
        busy |= {os.path.abspath(s) for s in ENGINE.current.sources}
    for e in files[KEEP_UPLOADS:]:
        if os.path.abspath(e.path) not in busy:
            try:
                os.remove(e.path)
            except OSError:
                pass


def list_runs():
    out = []
    try:
        names = sorted((e.name for e in os.scandir(RUNS_DIR) if e.is_dir()), reverse=True)
    except FileNotFoundError:
        return out
    live = {}                                   # runs the engine is working on right now: use their live state
    cur = ENGINE.current
    for r in ([cur] if cur else []) + list(ENGINE.queued):
        live[r.id] = snapshot(r.meta)
    for n in names:
        if n in live:
            m = live[n]
        else:
            try:
                with open(os.path.join(RUNS_DIR, n, "run.json"), encoding="utf-8") as f:
                    m = json.load(f)
            except (OSError, ValueError):
                continue
            if m.get("status") in ("queued", "running"):    # the server stopped while this run was going
                m["status"] = "stopped"
        out.append({"id": m["id"], "kind": m["kind"], "name": m["name"], "status": m["status"],
                    "created": m["created"], "plates": len(m["events"]) if m["kind"] == "video"
                    else sum(p["plate_like"] for e in m["photos"] for p in e["plates"])})
    return out


def run_meta(rid):
    cur = ENGINE.current
    if cur and cur.id == rid:
        return snapshot(cur.meta)
    for r in list(ENGINE.queued):
        if r.id == rid:
            return snapshot(r.meta)
    with open(os.path.join(RUNS_DIR, rid, "run.json"), encoding="utf-8") as f:
        m = json.load(f)
    if m.get("status") in ("queued", "running"):         # the server stopped while this run was going
        m["status"] = "stopped"
    return m


def run_dir(rid):
    d = os.path.realpath(os.path.join(RUNS_DIR, rid))
    if os.path.dirname(d) != os.path.realpath(RUNS_DIR) or not os.path.isdir(d):
        raise FileNotFoundError(rid)
    return d


class Handler(BaseHTTPRequestHandler):
    server_version = "PlateApp/1.0"

    def log_message(self, fmt, *a):      # keep the terminal for plate announcements
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, download=False):
        if not os.path.isfile(path):
            return self.send_error(HTTPStatus.NOT_FOUND)
        size = os.path.getsize(path)
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes=") and size:
            a, _, b = rng[6:].partition("-")
            try:
                if a == "":                                    # "the last N bytes"
                    start = max(0, size - int(b))
                else:
                    start = int(a)
                    end = min(size - 1, int(b)) if b else size - 1
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    return self.end_headers()
                code = 206
            except ValueError:
                start, end = 0, size - 1
        self.send_response(code)
        self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(path)}"')
        self.end_headers()
        left = end - start + 1
        with open(path, "rb") as f:
            f.seek(start)
            while left > 0:
                chunk = f.read(min(left, 1 << 20))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        p = url.path
        try:
            if p in ("/", "/index.html"):
                return self._file(os.path.join(HERE, "plate_app.html"))
            if p == "/api/state":
                cur = ENGINE.current
                return self._json({"current": snapshot(cur.meta) if cur else None,
                                   "queued": [r.meta["name"] for r in ENGINE.queued],
                                   "model": os.path.basename(ENGINE.lprnet_ckpt), "device": str(ENGINE.dev)})
            if p == "/api/runs":
                return self._json(list_runs())
            if p.startswith("/api/run/"):
                return self._json(run_meta(safe_name(p[len("/api/run/"):])))
            if p == "/live.mjpg":
                return self._mjpeg()
            if p.startswith("/runs/"):
                rid, _, rel = p[len("/runs/"):].partition("/")
                d = run_dir(urllib.parse.unquote(rid))
                f = os.path.realpath(os.path.join(d, urllib.parse.unquote(rel)))
                if not f.startswith(d + os.sep):
                    return self.send_error(HTTPStatus.FORBIDDEN)
                return self._file(f, download="download" in urllib.parse.parse_qs(url.query))
            self.send_error(HTTPStatus.NOT_FOUND)
        except (FileNotFoundError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with ENGINE.lock:
            ENGINE.viewers += 1
        seq = -1
        try:
            while True:
                with ENGINE.frame_cond:
                    ENGINE.frame_cond.wait_for(lambda: ENGINE.frame_seq != seq, timeout=2)
                    seq, jpg = ENGINE.frame_seq, ENGINE.frame_jpg
                if ENGINE.current is None and jpg is None:
                    break
                if jpg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with ENGINE.lock:
                ENGINE.viewers -= 1

    def do_POST(self):
        p = urllib.parse.urlparse(self.path).path
        try:
            if p == "/api/upload":
                return self._upload()
            if p == "/api/start":
                return self._start(self._body_json())
            if p == "/api/stop":
                ENGINE.stop_current()
                return self._json({"ok": True})
            if p == "/api/open":
                return self._open(self._body_json())
            self.send_error(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, KeyError, TypeError) as e:       # a malformed request must not kill the connection silently
            try:
                self._json({"error": f"bad request: {e}"}, 400)
            except OSError:
                pass

    def _upload(self):
        """The browser sends the picked file as the raw request body (no multipart parsing needed)."""
        name = safe_name(urllib.parse.unquote(self.headers.get("X-File-Name", "upload")))
        bare = os.path.splitext(name)[1] == ""          # no extension: decide from the content once it is saved
        if not bare and not kind_of(name):
            return self._json({"error": f"{name}: not a video or photo file"}, 400)
        left = int(self.headers.get("Content-Length") or 0)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        k = 1
        dest = os.path.join(UPLOAD_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_{k}_{name}")
        while os.path.exists(dest):
            dest = os.path.join(UPLOAD_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_{k}_{name}")
            k += 1
        with open(dest, "wb") as f:
            while left > 0:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left:
            os.remove(dest)
            return self._json({"error": "upload interrupted"}, 400)
        if bare:
            k = sniff_kind(dest)
            if k is None:
                os.remove(dest)
                return self._json({"error": f"{name}: not a video or photo file"}, 400)
            os.replace(dest, dest + (".mov" if k == "video" else ".jpg"))
            dest += ".mov" if k == "video" else ".jpg"
        return self._json({"path": dest, "name": name})

    def _start(self, req):
        st = req.get("settings") or {}
        settings = {"mode": "all" if st.get("mode") == "all" else "live",
                    "imgsz": str(st.get("imgsz", "auto")) if str(st.get("imgsz", "auto")) in ("auto", "640", "1280")
                    else "auto",
                    "record": bool(st.get("record", True)),
                    "compare": bool(st.get("compare", True)),
                    "workers": int(st.get("workers", 1)) if str(st.get("workers", 1)) in ("1", "2", "3", "4", "6", "8") else 1}
        paths = []
        for u in req.get("uploads", []):          # files uploaded through /api/upload
            f = os.path.realpath(os.path.join(UPLOAD_DIR, os.path.basename(u)))
            if os.path.isfile(f):
                paths.append(f)
        for raw in req.get("paths", []):          # paths typed/pasted on the page
            f = to_wsl_path(raw)
            if os.path.isdir(f):
                paths += [os.path.join(f, n) for n in sorted(os.listdir(f), key=natural_key) if kind_of(os.path.join(f, n))]
            elif os.path.isfile(f):
                paths.append(f)
            else:
                return self._json({"error": f"not found: {raw}"}, 400)
        videos = [f for f in paths if kind_of(f) == "video"]
        photos = [f for f in paths if kind_of(f) == "photo"]
        if not videos and not photos:
            return self._json({"error": "no video or photo files given"}, 400)
        runs = [Run("video", [v], settings) for v in videos]
        if photos:
            runs.append(Run("photo", photos, settings))
        for r in runs:
            ENGINE.submit(r)
        return self._json({"runs": [r.id for r in runs]})

    def _open(self, req):
        """Opens a run's folder or recording on the Windows desktop (Explorer / default video player)."""
        d = run_dir(safe_name(req.get("run", "")))
        target = os.path.join(d, "annotated.mp4") if req.get("what") == "recording" else d
        if not os.path.exists(target):
            return self._json({"error": "not found"}, 404)
        try:
            win = subprocess.run(["wslpath", "-w", target], capture_output=True, text=True, check=True).stdout.strip()
            subprocess.Popen(["explorer.exe", win], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError) as e:
            return self._json({"error": f"could not open: {e}"}, 500)
        return self._json({"ok": True, "path": win})


ENGINE = None


def main():
    global ENGINE
    defaults = build_parser().parse_args(["--source", "x"])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 = only this computer (default); 0.0.0.0 = also other devices on your network")
    ap.add_argument("--lprnet_ckpt", default=defaults.lprnet_ckpt)
    ap.add_argument("--yolo_ckpt", default=defaults.yolo_ckpt)
    args = ap.parse_args()
    os.chdir(HERE)                 # the default checkpoint paths are relative to uvm/
    os.makedirs(RUNS_DIR, exist_ok=True)
    print("loading models ...", flush=True)
    ENGINE = Engine(args.lprnet_ckpt, args.yolo_ckpt)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"\n  Plate reader is running:  open  http://localhost:{args.port}  in your browser"
          f"\n  results are saved in {RUNS_DIR}\n  Ctrl+C here to stop\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopping")


if __name__ == "__main__":
    main()
