"""
Extracts frames from traffic videos and PRE-LABELS every plate it can find, so a person only has to
correct the boxes (delete wrong ones, draw the few missed ones) instead of drawing everything.

Why pre-labels need a human check: training only on the current detector's own boxes would teach it
that every plate it currently misses (small bike/auto plates) is background.

Pre-labels = current detector at 640 AND 1280 px, low confidence, merged (high recall on purpose:
deleting a wrong box is quicker than drawing a missed one).

Output (YOLO format, class 0 = plate):
  <out>/images/<video>_<frame>.jpg   full-resolution frames
  <out>/labels/<video>_<frame>.txt   pre-labels
  <out>/labels.txt                   class list for makesense.ai

Correct them in https://www.makesense.ai (free, runs in the browser, images stay on your PC):
  1. Get Started -> drop all files from <out>/images -> Object Detection
  2. create one label named  plate
  3. Actions -> Import Annotations -> YOLO -> select ALL files from <out>/labels plus labels.txt
  4. go through the frames: delete wrong boxes, draw every readable-or-not plate you can see
  5. Actions -> Export Annotations -> YOLO  (zip) -> unzip into <out>/labels_fixed

Run from WSL2:
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/yolo_plate_detection
    python make_det_frames.py --videos 1 2 3 5 --out ../datasets/det_train_frames
    python make_det_frames.py --videos 4 --out ../datasets/det_test_frames
"""
import argparse
import os
import sys

import cv2
import torch
from PIL import Image
from torchvision.ops import nms
from ultralytics import YOLO

DEFAULT_CKPT = "runs/detect/runs/detect/anpr_plate_model_15k/weights/best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True, help="video numbers inside --video_dir, e.g. 1 2 3")
    ap.add_argument("--video_dir", default="../datasets/dataset14")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every_sec", type=float, default=2.0, help="one frame every N seconds")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--min_px", type=int, default=10, help="drop boxes narrower than this")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    args = ap.parse_args()

    yolo = YOLO(args.ckpt)
    os.makedirs(f"{args.out}/images", exist_ok=True)
    os.makedirs(f"{args.out}/labels", exist_ok=True)
    with open(f"{args.out}/labels.txt", "w") as f:
        f.write("plate\n")

    total_frames = total_boxes = 0
    for v in args.videos:
        cap = cv2.VideoCapture(os.path.join(args.video_dir, f"{v}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(fps * args.every_sec)))
        for pos in range(step // 2, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok:
                continue
            H, W = frame.shape[:2]
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, scores = [], []
            for sz in (640, 1280):
                for r in yolo.predict(img, conf=args.conf, imgsz=sz, verbose=False):
                    boxes += r.boxes.xyxy.tolist()
                    scores += r.boxes.conf.tolist()
            lines = []
            if boxes:
                b, s = torch.tensor(boxes), torch.tensor(scores)
                for i in nms(b, s, 0.5).tolist():
                    x1, y1, x2, y2 = b[i].tolist()
                    if x2 - x1 < args.min_px:
                        continue
                    lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            name = f"v{v}_{pos:05d}"
            img.save(f"{args.out}/images/{name}.jpg", quality=95)
            with open(f"{args.out}/labels/{name}.txt", "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            total_frames += 1
            total_boxes += len(lines)
        cap.release()
        print(f"video {v}: done", flush=True)
    print(f"{total_frames} frames, {total_boxes} pre-labelled boxes -> {args.out}")


if __name__ == "__main__":
    main()
