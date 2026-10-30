"""
Step 1 of the real-only UVM training: extract frames from every real video and organise the real photos.

Inputs  (all real, nothing synthetic):
  datasets/newtraingopro/{train_video,test_video}/{hazy_video,clear_video}/N_*_video.mp4      GoProHazy: real fog / clear road videos (1920x1080, 30 fps)
  datasets/gdrive_extra_photos/*.jpg, *.mp4                                                    your Ponmudi photos and videos
  datasets/gopro_hazy/match/N.txt                                                              the DVD authors' hazy-frame -> clear-frame matches (train videos)

Output  datasets/real_scratch/
  frames/gopro/{train,test}/{hazy,clear}/N_FFFFF.jpg      every STEP-th video frame (JPEG q97, 4:4:4; the videos are already lossy H.264, --format png for PNG)
  frames/ponmudi_photo/*.jpg, frames/ponmudi_video/*.jpg  upright copies / every VSTEP-th frame
  gopro_train_pairs.csv   hazy_path, clear_path, video, hazy_frame, clear_frame   (clear frame = the place-matched one, NOT pixel aligned)
  gopro_val_pairs.csv     held-out pairs from the test videos (never trained on)
  ponmudi_hazy.csv / ponmudi_clear.csv   real Ponmudi foggy images / real Ponmudi clear images (split by the haze level of each image)

    python real_scratch/extract_frames.py --step 2            (run from uvm/; add --limit_videos 2 --step 30 for a quick test)
"""
import argparse
import csv
import glob
import os
import re
import sys
import threading
import time
from multiprocessing import Pool, Value

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "datasets")
ap = argparse.ArgumentParser()
ap.add_argument("--gopro", default=os.path.join(ROOT, "newtraingopro"))
ap.add_argument("--extra", default=os.path.join(ROOT, "gdrive_extra_photos"))
ap.add_argument("--match", default=os.path.join(ROOT, "gopro_hazy", "match"))
ap.add_argument("--out", default=os.path.join(ROOT, "real_scratch"))
ap.add_argument("--step", type=int, default=2, help="keep every STEP-th frame of the GoPro videos (30 fps; 2 = 15 fps, 1 = every frame)")
ap.add_argument("--vstep", type=int, default=6, help="keep every VSTEP-th frame of your Ponmudi videos (they are 30-60 fps)")
ap.add_argument("--format", default="jpg", choices=["jpg", "png"])
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--limit_videos", type=int, default=0, help="only the first N GoPro videos per folder (for a quick test)")
ap.add_argument("--train_on_test", action="store_true", help="also train on the test_video pairs (by default they stay held out for the per-epoch check)")
ap.add_argument("--clear_max", type=float, default=0.30, help="Ponmudi image with dark-channel haze <= this is a CLEAR image")
ap.add_argument("--hazy_min", type=float, default=0.33, help="Ponmudi image with haze >= this is a FOGGY image (between = left out)")
a = ap.parse_args()
EXT = a.format


def save(img_bgr, path):
    rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if EXT == "jpg":
        rgb.save(path, quality=97, subsampling=0)
    else:
        rgb.save(path, compress_level=3)


def haze_level(bgr):
    s = 768 / max(bgr.shape[:2])
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
    return float(cv2.erode(small.astype(np.float32).min(2) / 255.0, np.ones((15, 15), np.uint8)).mean())


COUNTER = None


def _init(c):
    global COUNTER
    COUNTER = c


def extract_video(job):
    path, outdir, step, prefix = job
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(path)
    f, saved = 0, []
    while True:
        ok = cap.grab()
        if not ok:
            break
        if f % 20 == 0 and COUNTER is not None:
            with COUNTER.get_lock():
                COUNTER.value += 20
        if f % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                name = "%s_%05d.%s" % (prefix, f, EXT)
                dst = os.path.join(outdir, name)
                if not os.path.exists(dst):
                    save(frame, dst)
                saved.append((f, dst))
        f += 1
    cap.release()
    return path, saved


def vid_no(p):
    return int(re.match(r"(\d+)_", os.path.basename(p)).group(1))


def read_match(n):
    p = os.path.join(a.match, "%d.txt" % n)
    if not os.path.exists(p):
        return None
    out = []
    for l in open(p, encoding="utf-8"):
        h, c = l.strip().split("|")
        out.append((int(re.search(r"frame_(\d+)_hazy", h).group(1)), int(re.search(r"frame_(\d+)_clear", c).group(1))))
    return dict(out)


if __name__ == "__main__":
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for split in ("train", "test"):
        for kind in ("hazy", "clear"):
            vids = sorted(glob.glob(os.path.join(a.gopro, split + "_video", kind + "_video", "*.mp4")), key=vid_no)
            if a.limit_videos:
                vids = vids[:a.limit_videos]
            for v in vids:
                n = vid_no(v)
                jobs.append((v, os.path.join(a.out, "frames", "gopro", split, kind), a.step, str(n)))
    for v in sorted(glob.glob(os.path.join(a.extra, "*.mp4"))):
        if a.limit_videos and len([j for j in jobs if "ponmudi_video" in j[1]]) >= a.limit_videos:
            break
        jobs.append((v, os.path.join(a.out, "frames", "ponmudi_video"), a.vstep, os.path.splitext(os.path.basename(v))[0]))
    print("videos to extract:", len(jobs), "| step", a.step, "| format", EXT, flush=True)
    frames = {}
    total_frames = 0
    for j in jobs:
        c0 = cv2.VideoCapture(j[0])
        total_frames += int(c0.get(cv2.CAP_PROP_FRAME_COUNT))
        c0.release()
    counter = Value("l", 0)
    bar = tqdm(total=total_frames, desc="extracting frames", unit="frame", dynamic_ncols=True, smoothing=0.05)
    done_videos = [0]
    stop = threading.Event()

    def watch():
        last = 0
        while not stop.is_set():
            v = counter.value
            bar.update(v - last)
            last = v
            bar.set_postfix(videos="%d/%d" % (done_videos[0], len(jobs)))
            time.sleep(0.5)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    with Pool(a.workers, initializer=_init, initargs=(counter,)) as pool:
        for i, (path, saved) in enumerate(pool.imap_unordered(extract_video, jobs)):
            frames[path] = saved
            done_videos[0] = i + 1
            tqdm.write("  done %s: %d frames kept" % (os.path.relpath(path, ROOT), len(saved)))
    stop.set()
    th.join()
    bar.update(max(0, total_frames - bar.n))
    bar.close()

    # ---- GoPro pairs: hazy frame -> place-matched clear frame (nearest extracted one)
    def rows_for(split):
        rows = []
        hz = sorted(p for p in frames if "/%s_video/hazy_video/" % split in p.replace("\\", "/"))
        for hp in hz:
            n = vid_no(hp)
            cp = [p for p in frames if "/%s_video/clear_video/" % split in p.replace("\\", "/") and vid_no(p) == n]
            if not cp:
                continue
            hf, cf = frames[hp], frames[cp[0]]
            if not hf or not cf:
                continue
            cidx = np.array([f for f, _ in cf])
            match = read_match(n) if split == "train" else None
            for f, hpath in hf:
                if match is not None:
                    k = int(round(f / 3.0))
                    ks = sorted(match)
                    k = min(ks, key=lambda x: abs(x - k))
                    target = 3 * match[k]
                else:                                      # test videos: no match file -> same relative position in the clear video
                    target = int(round(f / max(hf[-1][0], 1) * cf[-1][0]))
                j = int(np.argmin(np.abs(cidx - target)))
                rows.append([hpath, cf[j][1], n, f, int(cidx[j])])
        return rows
    tr = rows_for("train")
    va_all = rows_for("test")
    va = va_all[::max(1, len(va_all) // 80)]
    if a.train_on_test:
        tr = tr + va_all
    for name, rows in (("gopro_train_pairs.csv", tr), ("gopro_val_pairs.csv", va)):
        with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["hazy_path", "clear_path", "video", "hazy_frame", "clear_frame"])
            w.writerows(rows)
    print("GoPro train pairs: %d (from %d videos) | held-out validation pairs: %d" % (len(tr), len({r[2] for r in tr}), len(va)), flush=True)

    # ---- Ponmudi photos + video frames: upright copies, split into foggy / clear by measured haze
    hazy_rows, clear_rows, gap = [], [], 0
    pdir = os.path.join(a.out, "frames", "ponmudi_photo")
    os.makedirs(pdir, exist_ok=True)
    items = []
    for f in sorted(glob.glob(os.path.join(a.extra, "*.jpg")) + glob.glob(os.path.join(a.extra, "*.JPG"))):
        dst = os.path.join(pdir, os.path.splitext(os.path.basename(f))[0].replace("(", "_").replace(")", "") + "." + EXT)
        im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        if not os.path.exists(dst):
            im.save(dst, quality=97, subsampling=0) if EXT == "jpg" else im.save(dst, compress_level=3)
        items.append((dst, "photo", haze_level(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))))
    for p, saved in frames.items():
        if "ponmudi_video" in saved[0][1].replace("\\", "/") if saved else False:
            for f, dst in saved:
                items.append((dst, "video", haze_level(cv2.imread(dst))))
    for path, src, h in tqdm(items, desc="measuring Ponmudi images", unit="img", dynamic_ncols=True):
        if h <= a.clear_max:
            clear_rows.append([path, src, round(h, 3)])
        elif h >= a.hazy_min:
            hazy_rows.append([path, src, round(h, 3)])
        else:
            gap += 1
    for name, rows in (("ponmudi_hazy.csv", hazy_rows), ("ponmudi_clear.csv", clear_rows)):
        with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "source", "haze"])
            w.writerows(rows)
    print("Ponmudi real images: %d foggy (%d photos + %d video frames) | %d clear | %d in between (not used)" % (
        len(hazy_rows), sum(r[1] == "photo" for r in hazy_rows), sum(r[1] == "video" for r in hazy_rows), len(clear_rows), gap), flush=True)
    print("EXTRACT_FINISHED")
