"""
Fine-tunes the plate detector for traffic video (many small bike/auto plates), starting from the
current 15k-image model.

Why: the 15k training set is almost all close-ups (median plate 180 px wide, 5% under 40 px), so the
model finds only about half of the small plates in our traffic frames (eval_detector.py).

Training data (copied into the WSL filesystem first, because reading thousands of files over /mnt/c
is very slow):
  * our corrected traffic frames (det_train_frames, labels_fixed), each copied --video_repeat times
  * a random --base_n subset of the original 15k set, so the model keeps what it already knows
  * the 525-image YOLO_dataset (14% small plates)
Monitoring/val: YOLO_dataset val (133 images). The real test is eval_detector.py on det_test_frames
(video 4), which is never trained on. The final model is last.pt after a fixed number of epochs.

Run from WSL2:
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/yolo_plate_detection
    python finetune_traffic.py build
    nohup python finetune_traffic.py train > finetune_traffic.log 2>&1 &
"""
import argparse
import os
import random
import shutil

HOME = os.path.expanduser("~/yolo_ft")
VIDEO = "/mnt/c/projects/jewwww/datasets/det_train_frames"
BASE = "/mnt/c/anpr15k/License Plate Detection/train"
SMALL = "/mnt/c/projects/jewwww/yolo_plate_detection/data/YOLO_dataset"
START = "/mnt/c/projects/jewwww/yolo_plate_detection/runs/detect/runs/detect/anpr_plate_model_15k/weights/best.pt"


def copy_pair(img_src, lab_src, split, name):
    os.makedirs(f"{HOME}/{split}/images", exist_ok=True)
    os.makedirs(f"{HOME}/{split}/labels", exist_ok=True)
    ext = os.path.splitext(img_src)[1].lower()
    shutil.copyfile(img_src, f"{HOME}/{split}/images/{name}{ext}")
    if lab_src and os.path.exists(lab_src):
        shutil.copyfile(lab_src, f"{HOME}/{split}/labels/{name}.txt")
    else:
        open(f"{HOME}/{split}/labels/{name}.txt", "w").close()   # image with no plates


def build(args):
    if os.path.exists(HOME):
        shutil.rmtree(HOME)
    # 1. our traffic frames, repeated so they carry real weight next to the big base set
    labs = sorted(f for f in os.listdir(f"{VIDEO}/labels_fixed") if f.endswith(".txt") and f != "labels.txt")
    if not labs:
        raise SystemExit(f"no corrected labels in {VIDEO}/labels_fixed -- export them from makesense.ai first")
    for f in labs:
        stem = f[:-4]
        for r in range(args.video_repeat):
            copy_pair(f"{VIDEO}/images/{stem}.jpg", f"{VIDEO}/labels_fixed/{f}", "train", f"vid_{stem}_r{r}")
    # 2. subset of the original 15k set
    base_imgs = os.listdir(f"{BASE}/images")
    for f in random.Random(0).sample(base_imgs, min(args.base_n, len(base_imgs))):
        stem = os.path.splitext(f)[0]
        copy_pair(f"{BASE}/images/{f}", f"{BASE}/labels/{stem}.txt", "train", f"base_{stem}")
    # 3. the small-plate set (train) and its val split for monitoring
    for split_src, split_dst, prefix in (("train", "train", "small"), ("val", "val", "smallval")):
        for f in os.listdir(f"{SMALL}/images/{split_src}"):
            stem = os.path.splitext(f)[0]
            copy_pair(f"{SMALL}/images/{split_src}/{f}", f"{SMALL}/labels/{split_src}/{stem}.txt", split_dst, f"{prefix}_{stem}")
    with open(f"{HOME}/data.yaml", "w") as fh:
        fh.write(f"path: {HOME}\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['plate']\n")
    n_train = len(os.listdir(f"{HOME}/train/images"))
    n_val = len(os.listdir(f"{HOME}/val/images"))
    print(f"traffic frames: {len(labs)} x {args.video_repeat} | base subset: {args.base_n} | small set: "
          f"{len(os.listdir(SMALL + '/images/train'))} -> {n_train} training images, {n_val} val images in {HOME}")


def train(args):
    from ultralytics import YOLO
    model = YOLO(START)
    model.train(data=f"{HOME}/data.yaml", imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                lr0=0.002, lrf=0.1, warmup_epochs=1, close_mosaic=5, patience=0, seed=0, workers=4,
                device=0, project="/mnt/c/projects/jewwww/yolo_plate_detection/runs_ft", name=args.name,
                exist_ok=True, plots=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "train"])
    ap.add_argument("--video_repeat", type=int, default=8)
    ap.add_argument("--base_n", type=int, default=3000)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="traffic_1280")
    args = ap.parse_args()
    build(args) if args.step == "build" else train(args)


if __name__ == "__main__":
    main()
