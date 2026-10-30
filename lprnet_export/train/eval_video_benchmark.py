"""
Scores LPRNet checkpoints on plates from real traffic videos, using crops saved once by
    read_video.py --save_track_crops 3 ...
so every checkpoint is judged on exactly the same frames, without re-running detection.

The answer key is ../datasets/video_benchmark.json: plates read BY EYE from enlarged crops
(uncertain ones left out). None of these videos were used for training.

For each plate it reports whether the track vote (same rule as read_video.py) gives the right
text, plus the share of individual frames read exactly.

    python eval_video_benchmark.py ckpt1.pt [ckpt2.pt ...] [--show_diff]
"""
import argparse
import glob
import json
import os

import torch
from PIL import Image

from model_lprnet import LPRNet
from read_plate import read_plate
from read_video import Track

D = "../datasets"


def load_benchmark(path, videos=None):
    bench = json.load(open(path, encoding="utf-8-sig"))
    items = []
    for e in bench["plates"]:
        if videos and e["video_out"] not in videos:
            continue
        files = []
        for t in e["tracks"]:
            files += sorted(glob.glob(os.path.join(D, e["video_out"], "track_crops", str(t), "*.png")))
        if files:
            items.append((e["video_out"], e["plate"], [Image.open(f).convert("RGB") for f in files]))
    return items


@torch.no_grad()
def score(ckpt, items, dev):
    net = LPRNet(num_classes=37).to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev))
    net.eval()
    votes, frames_ok, frames_all = {}, 0, 0
    for vid, plate, crops in items:
        t = Track(0, (0, 0, 1, 1), 0)
        for c in crops:
            text = read_plate(net, c, dev)[1]
            t.readings.append((text, c.width * c.height))
            frames_ok += text == plate
            frames_all += 1
        votes[(vid, plate)] = t.consensus()[0]
    return votes, frames_ok / max(frames_all, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--benchmark", default=os.path.join(D, "video_benchmark.json"))
    ap.add_argument("--show_diff", action="store_true", help="list plates where the checkpoints disagree")
    ap.add_argument("--videos", nargs="+", default=None,
                    help="only these video output folders, e.g. dataset14_out4 (videos 2-3 plates are in the "
                         "video_ocr_crops training data since 2026-10-04, so only video 4 is a fair test)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = load_benchmark(args.benchmark, args.videos)
    print(f"{len(items)} plates, {sum(len(c) for *_, c in items)} saved frames\n")
    results = {}
    for ck in args.ckpts:
        votes, frame_acc = score(ck, items, dev)
        results[ck] = votes
        ok = sum(votes[(v, p)] == p for v, p, _ in items)
        print(f"{ck:<60} plates right {ok:>3}/{len(items)} ({ok / len(items):.0%})   frames right {frame_acc:.1%}")
    if args.show_diff and len(args.ckpts) > 1:
        print("\nplates where the checkpoints disagree (truth: reading per checkpoint):")
        for v, p, _ in items:
            reads = [results[ck][(v, p)] for ck in args.ckpts]
            if len(set(reads)) > 1:
                print(f"  {p:<12} " + "  ".join(f"{r}{'*' if r == p else ''}" for r in reads))


if __name__ == "__main__":
    main()
