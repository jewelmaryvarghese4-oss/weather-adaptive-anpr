"""
Count how many plates a checkpoint reads exactly right on a test folder that has labels.json and crops/.
  python evaluate.py --test_dir data/foggy_crops_test
  python evaluate.py --test_dir data/heldout_rj250 --ckpt weights/weather_trained/lprnet_weather_seed1.pt
Prints exact matches of the plain reading and of the format-corrected reading.
  --view all     scores the three ways of reading side by side: plain crop | smoothed view | choose the better of the two
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from PIL import Image

from predict_plate import load_net, read_plate, read_plate_view

ap = argparse.ArgumentParser()
ap.add_argument("--test_dir", required=True)
ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "lprnet_avg_INSTALLED.pt"))
ap.add_argument("--show", action="store_true", help="print every plate")
ap.add_argument("--view", choices=["plain", "smooth", "select", "all"], default="plain")
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = load_net(a.ckpt, dev)
labels = json.load(open(os.path.join(a.test_dir, "labels.json")))
views = ["plain", "smooth", "select"] if a.view == "all" else [a.view]
raw_ok = {v: 0 for v in views}
fmt_ok = {v: 0 for v in views}
for pid, truth in labels.items():
    img = Image.open(os.path.join(a.test_dir, "crops", pid + ".png"))
    for v in views:
        raw, fmt, _, _ = read_plate_view(net, img, dev, v)
        raw_ok[v] += raw == truth
        fmt_ok[v] += fmt == truth
        if a.show:
            print("%-14s %-7s truth=%-12s plain=%-12s corrected=%-12s %s" % (pid, v, truth, raw, fmt, "OK" if fmt == truth else ""))
for v in views:
    print("%s | %d plates | view %-6s | exact (plain) %d | exact (format-corrected) %d" % (os.path.basename(a.ckpt), len(labels), v, raw_ok[v], fmt_ok[v]))
