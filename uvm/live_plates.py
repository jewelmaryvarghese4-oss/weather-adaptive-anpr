"""
REAL-TIME plate reading: shows the video with plate boxes live, and the moment a vehicle's plate is
confirmed it appears on screen and is printed with its time, e.g.
    [  12.4 s]  KL01AB8297   (track 7)

Same detector -> tracker -> LPRNet -> vote pipeline as read_video.py, rebuilt for speed:
  * the vote is updated only when a new reading arrives, over the most frequent readings only
    (read_video.py recomputes it from scratch every frame, which grows with how long a car stays)
  * each plate is read every --ocr_every frames, and no longer once its reading is confirmed
  * YOLO runs once per frame, in half precision, directly on the camera/video frame
  * nothing is written to disk unless you ask (--save)
  * --realtime (default for files): if processing falls behind, frames are skipped so what you see
    is always "now", like a live camera

Sources: a video file, a camera number (0 = first webcam) or a stream URL (rtsp://...).

Run from WSL2 (a window opens on your Windows desktop; press q in it to stop):
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python live_plates.py --source ../datasets/dataset14/4.mp4
"""
import argparse
import collections
import csv
import os
import time

import cv2
import torch
from PIL import Image
from ultralytics import YOLO

from async_writer import AsyncWriter
from lprnet_vocab import NUM_CLASSES
from model_lprnet import LPRNet
from read_plate import DEFAULT_YOLO_CKPT, looks_like_plate, read_plate
from read_video import (MIN_SUPPORT, MIN_SUPPORT_SHARE, STRICT_MIN_RATIO, STRICT_PLATE, center_dist, iou,
                        levenshtein)

TOP_K = 12          # the vote only considers the 12 most frequent distinct readings of a track


class LiveTrack:
    def __init__(self, tid, box, frame_idx):
        self.id, self.box, self.last = tid, box, frame_idx
        self.weight = collections.Counter()   # reading -> summed weight (conf x crop area)
        self.freq = collections.Counter()     # reading -> how many reads gave it
        self.reads = 0
        self.last_read = -10 ** 9
        self.text, self.reliable = "", False
        self.history = collections.deque(maxlen=5)   # consensus after each of the last reads
        self.frozen = False                   # confirmed and stable: stop reading it
        self.announced = False
        self.best_crop, self.best_area = None, 0   # largest crop that was read (for --save_crops)
        self.first_idx = frame_idx
        self.seen = 0                         # frames in which the detector found this plate
        self.max_w = 0

    def add_reading(self, text, weight):
        self.reads += 1
        if text:
            self.weight[text] += weight
            self.freq[text] += 1
        self._vote()
        self.history.append(self.text)

    def _vote(self):
        """Same rules as read_video.Track.consensus(), restricted to the TOP_K most frequent readings."""
        top = [t for t, _ in self.freq.most_common(TOP_K)]
        strict = [t for t in top if STRICT_PLATE.match(t)]
        good = [t for t in top if looks_like_plate(t)]
        if strict and good and self.freq[strict[0]] >= STRICT_MIN_RATIO * self.freq[good[0]]:
            pool = strict
        else:
            pool = good or strict or top
        if not pool:
            self.text, self.reliable = "", False
            return
        by_freq = sorted(pool, key=lambda c: -self.freq[c])
        if len(by_freq) == 1 or self.freq[by_freq[0]] >= 1.5 * self.freq[by_freq[1]]:
            self.text = by_freq[0]          # one reading clearly dominates: trust it
        else:                               # otherwise the reading closest to all the others
            self.text = min(pool, key=lambda c: (sum(self.weight[o] * levenshtein(c, o) for o in pool), -self.weight[c]))
        sup = self.freq[self.text]
        self.reliable = looks_like_plate(self.text) and sup >= MIN_SUPPORT and sup >= MIN_SUPPORT_SHARE * self.reads

    def needs_read(self, idx, every, early=10):
        # read EVERY frame while a plate is new: a plate that passes quickly may only be visible for ~10
        # frames and would otherwise collect too few reads to be confirmed
        return not self.frozen and (self.reads < early or idx - self.last_read >= every)

    def passes_on_exit(self):
        """A plate that left view without being confirmed: still announce it if it is a clear, well-formed
        plate that most of its reads agree on (short-lived plates never reach the normal 5-read rule)."""
        sup = self.freq[self.text] if self.text else 0
        return bool(self.text and STRICT_PLATE.match(self.text) and sup >= 3 and sup >= 0.3 * self.reads)


class LiveReader:
    """Detector -> tracker -> LPRNet -> vote, one frame at a time. Driven by main() below and by plate_app.py
    (the browser interface). on_announce(track, video_time, old_plate) is called for every new plate
    (old_plate None) and every correction of an announced one."""

    def __init__(self, lprnet, yolo, dev, fps, args, keep_crops=False, on_announce=None):
        self.lprnet, self.yolo, self.dev, self.fps, self.args = lprnet, yolo, dev, fps, args
        self.half = dev.type == "cuda"
        self.keep_crops = keep_crops     # remember each track's largest crop (best_crop)
        self.on_announce = on_announce
        self.active, self.next_id = [], 1
        self.all_tracks = {}
        self.export_reads = []           # (track, frame, crop w, crop h, text) for --export_run
        self.events = []                 # (video time s, plate, track id)
        self.recent = collections.deque(maxlen=8)
        self.last_seen = {}              # announced plate -> video time it was last on screen
        self.n_ocr = 0
        self.samples = collections.defaultdict(list)   # track id -> [(frame idx, box)] where it was seen (for the smooth annotated video)
        self.prev_idx, self.gap = None, 1.0            # how far apart the frames given to update() are (1 = every frame)

    def announce(self, tr, vt):
        # one announcement per vehicle: skip if the same/near-same plate was still on screen in the
        # last 10 s (a vehicle that waits in traffic often loses and regains its track)
        for p0, t_seen in self.last_seen.items():
            if vt - t_seen < 10 and levenshtein(p0, tr.text) <= 1:
                tr.announced = tr.text
                return
        self.events.append((vt, tr.text, tr.id))
        self.recent.appendleft((vt, tr.text))
        tr.announced = tr.text
        print(f"[{vt:7.1f} s]  {tr.text:<12} (track {tr.id})", flush=True)
        if self.on_announce:
            self.on_announce(tr, vt, None)

    def correct(self, tr, vt):
        # the vote changed after the plate was announced: replace the earlier announcement
        old = tr.announced
        for k, (t0, p0, i0) in enumerate(self.events):
            if i0 == tr.id:
                self.events[k] = (t0, tr.text, i0)
        for k, (t0, p0) in enumerate(self.recent):
            if p0 == old:
                self.recent[k] = (t0, tr.text)
        tr.announced = tr.text
        print(f"[{vt:7.1f} s]  {tr.text:<12} (track {tr.id}, corrected from {old})", flush=True)
        if self.on_announce:
            self.on_announce(tr, vt, old)

    def detect(self, frame):
        """Finds the plates in one frame (YOLO only): [(box, confidence)]."""
        args = self.args
        dets = []
        for r in self.yolo.predict(frame, conf=args.yolo_conf, imgsz=args.yolo_imgsz, half=self.half, verbose=False):
            for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                if b[2] - b[0] >= args.min_w and b[3] - b[1] >= args.min_h:
                    dets.append((tuple(b), c))
        return dets

    def step(self, frame, idx):
        """Detects, tracks and reads the plates of frame number idx (BGR, as read by OpenCV), all in this process."""
        dets = self.detect(frame)

        def read(di, box):
            x1, y1, x2, y2 = (int(v) for v in box)
            crop = Image.fromarray(cv2.cvtColor(frame[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2RGB))
            _, text = read_plate(self.lprnet, crop, self.dev)
            return crop, text

        self.update(dets, idx, read)

    def update(self, dets, idx, read):
        """Tracks the plates found in frame idx and updates the votes. dets = [(box, conf)]; read(i, box) returns
        (crop as a PIL image, plate text) for detection i and is called only for plates that still need a reading (so the
        single-process path reads as little as before, while parallel_reader's workers have already read every crop)."""
        args, fps = self.args, self.fps
        vt = idx / fps
        if self.prev_idx is not None:
            self.gap = 0.8 * self.gap + 0.2 * (idx - self.prev_idx)
        self.prev_idx = idx
        # a plate is dropped after max_age frames without a detection; when frames are skipped the same number of video frames
        # holds fewer chances to see it, so the limit grows with the gap (stays max_age when every frame is processed)
        max_age = max(args.max_age, int(3 * self.gap))
        active = self.active
        pairs = sorted(((iou(t.box, b), ti, di) for ti, t in enumerate(active) for di, (b, _) in enumerate(dets)),
                       reverse=True)
        used_t, used_d, match = set(), set(), {}
        for sc, ti, di in pairs:
            b = dets[di][0]
            near = center_dist(active[ti].box, b) < 0.8 * max(b[2] - b[0], b[3] - b[1])
            if ti in used_t or di in used_d or (sc < 0.1 and not near):
                continue
            used_t.add(ti); used_d.add(di); match[di] = active[ti]

        for di, (box, conf) in enumerate(dets):
            tr = match.get(di)
            if tr is None:
                tr = LiveTrack(self.next_id, box, idx)
                self.next_id += 1
                active.append(tr)
                self.all_tracks[tr.id] = tr
            tr.box, tr.last = box, idx
            self.samples[tr.id].append((idx, box))
            tr.seen += 1
            tr.max_w = max(tr.max_w, box[2] - box[0])
            if tr.needs_read(idx, args.ocr_every, args.early_reads):
                crop, text = read(di, box)
                self.n_ocr += 1
                if args.export_run:
                    os.makedirs(os.path.join(args.export_run, "track_crops", str(tr.id)), exist_ok=True)
                    crop.save(os.path.join(args.export_run, "track_crops", str(tr.id), f"{idx:06d}.png"))
                    self.export_reads.append((tr.id, idx, crop.width, crop.height, text))
                if self.keep_crops and crop.width * crop.height > tr.best_area:
                    tr.best_crop, tr.best_area = crop, crop.width * crop.height
                tr.last_read = idx
                tr.add_reading(text, conf * crop.width * crop.height)
                if tr.reliable and not tr.announced:
                    self.announce(tr, vt)
                elif (tr.reliable and tr.announced and tr.text != tr.announced
                      and tr.freq[tr.text] >= max(1.5 * tr.freq[tr.announced], tr.freq[tr.announced] + 3)):
                    self.correct(tr, vt)   # only when the new reading clearly outnumbers the announced one
                if (tr.reliable and tr.freq[tr.text] >= args.freeze_after
                        and len(tr.history) == tr.history.maxlen and len(set(tr.history)) == 1):
                    tr.frozen = True
        for t in [t for t in active if idx - t.last > max_age]:     # plate has left the picture
            if not t.announced and t.passes_on_exit():
                self.announce(t, t.last / fps)
        self.active = [t for t in active if idx - t.last <= max_age]
        for t in self.active:
            if t.announced and t.last == idx:
                self.last_seen[t.announced] = vt

    def finish(self):
        """The video ended with these plates still in view."""
        for t in self.active:
            if not t.announced and t.passes_on_exit():
                self.announce(t, t.last / self.fps)

    def draw(self, frame, idx, proc_fps):
        """Plate boxes with their current reading, plus the list of the latest confirmed plates."""
        for t in self.active:
            if t.last != idx:
                continue
            x1, y1, x2, y2 = (int(v) for v in t.box)
            col = (0, 220, 0) if t.reliable else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
            if t.text:
                cv2.putText(frame, t.text, (x1, max(28, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (430, 60 + 38 * len(self.recent)), (0, 0, 0), -1)
        cv2.putText(frame, f"{idx / self.fps:6.1f} s   {proc_fps:4.0f} fps processing", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        for k, (t_ev, p) in enumerate(self.recent):
            cv2.putText(frame, f"{t_ev:6.1f} s  {p}", (12, 82 + 38 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0) if k == 0 else (200, 200, 200), 2, cv2.LINE_AA)


def load_models(lprnet_ckpt, yolo_ckpt):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lprnet = LPRNet(num_classes=NUM_CLASSES).to(dev)
    lprnet.load_state_dict(torch.load(lprnet_ckpt, map_location=dev))
    lprnet.eval()
    return lprnet, YOLO(yolo_ckpt), dev


def open_source(src):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"could not open source {src!r}")
    is_file = os.path.exists(src)
    return cap, is_file


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file, camera number (0) or stream URL")
    ap.add_argument("--lprnet_ckpt", default="checkpoints_lprnet/lprnet_avg.pt")
    ap.add_argument("--yolo_ckpt", default=DEFAULT_YOLO_CKPT)
    ap.add_argument("--yolo_imgsz", type=int, default=640)
    ap.add_argument("--yolo_conf", type=float, default=0.3)
    ap.add_argument("--min_w", type=int, default=24)
    ap.add_argument("--min_h", type=int, default=10)
    ap.add_argument("--ocr_every", type=int, default=3, help="read each plate every N processed frames")
    ap.add_argument("--early_reads", type=int, default=10,
                    help="read every frame for a plate's first N reads (short-lived plates need them)")
    ap.add_argument("--freeze_after", type=int, default=12,
                    help="stop reading a plate once this many reads agree and the vote is stable")
    ap.add_argument("--max_age", type=int, default=10)
    ap.add_argument("--no_realtime", action="store_true",
                    help="for files: process every frame even if slower than the video (default skips "
                         "frames to stay live)")
    ap.add_argument("--workers", type=int, default=1,
                    help="N > 1: N worker processes find and read the plates of different frames at the same time "
                         "(see parallel_reader.py), so fast or 4K video does not have to be thinned out. 1 = one process")
    ap.add_argument("--max_lag", type=float, default=1.0,
                    help="with --workers: only skip frames if the result runs more than this many seconds behind the video")
    ap.add_argument("--no_show", action="store_true", help="no window (e.g. for timing tests)")
    ap.add_argument("--display_width", type=int, default=1280)
    ap.add_argument("--save", default=None, help="optional path to save the annotated video (slower)")
    ap.add_argument("--save_width", type=int, default=0,
                    help="with --save: shrink the saved video to this width (e.g. 1920 for a 4K source, "
                         "which would otherwise be ~1 GB). 0 = native size")
    ap.add_argument("--log", default="live_plates_log.csv", help="CSV of confirmed plates with times")
    ap.add_argument("--export_run", default=None,
                    help="folder: export every tracked plate's crops + tracks.csv + readings.csv, ready for "
                         "video_label_review.py prepare / eval_video_benchmark.py")
    ap.add_argument("--tracks_csv", default=None,
                    help="optional CSV of EVERY tracked plate (confirmed or not) plus an unconfirmed_crops/ "
                         "folder next to it: for checking whether any vehicle was missed")
    ap.add_argument("--min_unconfirmed_frames", type=int, default=20,
                    help="with --tracks_csv: only save crops of unconfirmed tracks seen in at least this many frames")
    ap.add_argument("--save_crops", default=None,
                    help="optional folder: saves the clearest crop of every confirmed plate (<time>_<plate>.png)")
    ap.add_argument("--max_seconds", type=float, default=0, help="stop after this much video time (0 = all)")
    return ap


def main():
    args = build_parser().parse_args()
    if args.workers > 1:               # the workers hold the models; this process only decodes, tracks and draws
        lprnet = yolo = None
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        lprnet, yolo, dev = load_models(args.lprnet_ckpt, args.yolo_ckpt)

    cap, is_file = open_source(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    realtime = not args.no_realtime
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    save_scale = (args.save_width / W) if args.save_width and args.save_width < W else 1.0
    save_size = (int(W * save_scale), int(H * save_scale))
    writer = AsyncWriter(args.save, fps, save_size) if args.save else None

    reader = LiveReader(lprnet, yolo, dev, fps, args, keep_crops=bool(args.save_crops or args.tracks_csv))
    t_start = time.perf_counter()
    st = {"processed": 0, "skipped": 0, "t_start": t_start}
    proc_times = collections.deque(maxlen=30)

    def single_frames():
        """One process does everything: for files, jump ahead to the frame that "should" be showing now."""
        idx = -1                   # index of the frame in the source
        while True:
            if is_file and realtime and st["processed"]:
                target = int((time.perf_counter() - t_start) * fps)
                while idx + 1 < target:
                    if not cap.grab():
                        break
                    idx += 1
                    st["skipped"] += 1
            ok, frame = cap.read()
            if not ok:
                return
            idx += 1
            if args.max_seconds and idx / fps > args.max_seconds:
                return
            t0 = time.perf_counter()
            reader.step(frame, idx)
            st["processed"] += 1
            proc_times.append(time.perf_counter() - t0)
            yield idx, frame

    def parallel_frames():
        """Workers read different frames at the same time; results come back here in frame order."""
        from parallel_reader import WorkerPool, run_parallel
        print(f"starting {args.workers} workers ...", flush=True)
        pool = WorkerPool(args.workers, args.lprnet_ckpt, args.yolo_ckpt)
        t_prev = st["t_start"] = time.perf_counter()        # the speed is counted from here, not from the model loading
        try:
            for idx, frame, info in run_parallel(pool, cap, reader, args, fps, is_file=is_file, realtime=realtime,
                                                 max_lag=args.max_lag):
                if args.max_seconds and idx / fps > args.max_seconds:
                    return
                now = time.perf_counter()
                proc_times.append(now - t_prev)
                t_prev = now
                st["processed"] += 1
                st["skipped"] = info["skipped"]
                yield idx, frame
        finally:
            pool.close()

    idx = -1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    quit_early = False
    last_out = {"idx": -1, "frame": None}      # the last frame written to --save
    for idx, frame in (parallel_frames() if args.workers > 1 else single_frames()):
        # --- draw --------------------------------------------------------------------------------
        if not args.no_show or writer:
            reader.draw(frame, idx, len(proc_times) / max(sum(proc_times), 1e-6))
            if writer:
                out = cv2.resize(frame, save_size) if save_scale < 1 else frame.copy()
                # skipped frames: hold the previous picture so the video keeps its real speed
                writer.write(last_out["frame"] if last_out["frame"] is not None else out, idx - last_out["idx"] - 1)
                writer.write(out)
                last_out["idx"], last_out["frame"] = idx, out
            if not args.no_show:
                scale = args.display_width / W
                cv2.imshow("live plates  (q = quit)", cv2.resize(frame, (args.display_width, int(H * scale))))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    quit_early = True
                    break
    processed, skipped = st["processed"], st["skipped"]

    cap.release()
    reader.finish()
    if writer:
        writer.release()
        if st["skipped"] > 0 and is_file:
            # frames were skipped while running live: write a smooth version from the original frames (boxes on every frame)
            from render_video import render_smooth
            print("finishing the annotated video from the original frames ...", flush=True)
            smooth = args.save + ".smooth.mp4"
            end = idx if (quit_early or args.max_seconds or not total_frames) else max(idx, total_frames - 1)
            render_smooth(args.source, smooth, reader, fps, save_size, end)
            os.replace(smooth, args.save)
    cv2.destroyAllWindows()
    wall = time.perf_counter() - st["t_start"]
    events, all_tracks = reader.events, reader.all_tracks
    with open(args.log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_time_s", "plate", "track"])
        w.writerows([(f"{t:.1f}", p, i) for t, p, i in events])
    print(f"\n{processed} frames processed, {skipped} skipped to stay live | {wall:.1f} s wall time for "
          f"{idx / fps:.1f} s of video | processing {processed / wall:.1f} frames/s | "
          f"{reader.n_ocr / max(processed, 1):.2f} plate reads per frame")
    print(f"{len(events)} plates confirmed -> {args.log}")
    if args.export_run:
        # same layout as read_video.py --save_track_crops, so video_label_review.py / eval_video_benchmark.py
        # can use a live run directly: tracks.csv, readings.csv and track_crops/<track>/<frame>.png
        with open(os.path.join(args.export_run, "tracks.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "frames", "consensus", "plate_like", "top_readings", "support", "reliable"])
            for tid, t in sorted(all_tracks.items()):
                w.writerow([tid, t.seen, t.text, int(looks_like_plate(t.text)),
                            "; ".join(f"{s} x{c}" for s, c in t.freq.most_common(3)),
                            t.freq[t.text] if t.text else 0, int(t.reliable)])
        with open(os.path.join(args.export_run, "readings.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["track", "frame", "crop_w", "crop_h", "text"])
            w.writerows(reader.export_reads)
        print(f"exported {len(all_tracks)} tracks / {len(reader.export_reads)} reads -> {args.export_run}")
    if args.tracks_csv:
        # every tracked plate, announced or not -- to see which vehicles never got a confirmed plate
        os.makedirs(os.path.dirname(os.path.abspath(args.tracks_csv)), exist_ok=True)
        unconf_dir = os.path.join(os.path.dirname(os.path.abspath(args.tracks_csv)), "unconfirmed_crops")
        os.makedirs(unconf_dir, exist_ok=True)
        with open(args.tracks_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["track", "first_s", "last_s", "frames_seen", "reads", "max_width_px", "vote", "reliable",
                        "announced_as", "distinct_readings", "top_readings"])
            for tid, t in sorted(all_tracks.items()):
                w.writerow([tid, f"{t.first_idx / fps:.1f}", f"{t.last / fps:.1f}", t.seen, t.reads, int(t.max_w),
                            t.text, int(t.reliable), t.announced or "", len(t.freq),
                            "; ".join(f"{s} x{c}" for s, c in t.freq.most_common(3))])
                if not t.announced and t.seen >= args.min_unconfirmed_frames and t.best_crop is not None:
                    t.best_crop.save(os.path.join(unconf_dir, f"{t.first_idx / fps:06.1f}s_track{tid}_{t.seen}fr.png"))
        print(f"all tracks -> {args.tracks_csv}; clearest crops of long unconfirmed tracks -> {unconf_dir}")
    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)
        for t, p, i in events:
            tr = all_tracks.get(i)
            if tr is not None and tr.best_crop is not None:
                tr.best_crop.save(os.path.join(args.save_crops, f"{t:06.1f}s_{p}_track{i}.png"))
        print(f"clearest crop of each confirmed plate -> {args.save_crops}")


if __name__ == "__main__":
    main()
