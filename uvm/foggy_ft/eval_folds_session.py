"""
Scores each fold on the plates held out of that fold (datasets/foggy_ft/verified_fold{k}_test), three ways of reading (plain / smoothed view /
smoothed-or-plain chosen by the reader's confidence) for:
  installed | weather-only (no foggy crops) | previous fold model (verified foggy crops only) | NEW session fold model (+ session crops)
plus the new fold model on heldout_rj250 (clear plates).      python foggy_ft/eval_folds_session.py
"""
import json
import os
import re
import subprocess
import sys

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
UVM = os.path.dirname(HERE)
os.chdir(UVM)
sys.path.insert(0, UVM)
os.environ["TWO_LINE_MODE"] = "fixed"
import read_plate as rp
from model_lprnet import LPRNet

D = "../datasets"
dev = torch.device("cuda")
torch.set_grad_enabled(False)


def net_of(ck):
    n = LPRNet(num_classes=37).to(dev)
    n.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
    return n.eval()


def word(im):
    return im.resize((64, 16), Image.BICUBIC).resize((128, 32), Image.BICUBIC)


def read(net, img):
    lo, hi = rp.AMBIGUOUS_ASPECT
    cands = [img, rp.flatten_for_mode(img)] if lo <= img.width / img.height < hi else [rp.normalize_plate_layout(img)]
    best = None
    for c in cands:
        _, fmt, conf = rp._decode(net, c, dev)
        key = (rp.looks_like_plate(fmt), conf)
        if best is None or key > best[0]:
            best = (key, fmt, conf)
    return best[1], best[2], best[0][0]


def score(net, test_dir):
    labels = json.load(open(os.path.join(test_dir, "labels.json")))
    a = b = c = 0
    for pid, truth in labels.items():
        im = Image.open(os.path.join(test_dir, "crops", pid + ".png")).convert("RGB")
        ta, ca, va = read(net, im)
        tb, cb, vb = read(net, word(im))
        pick = max([(va, ca, ta), (vb, cb, tb)])[2]
        a += ta == truth
        b += tb == truth
        c += pick == truth
    return a, b, c, len(labels)


def clear(net):
    r = subprocess.run([sys.executable, "label_review.py", "eval", "--lprnet_ckpt", net, "--test_dir", D + "/heldout_rj250"],
                       capture_output=True, text=True, env={**os.environ}).stdout
    m = re.search(r"exact \(raw\) (\d+)/(\d+).*?format-corrected\) (\d+)/", r)
    return "%s exact / %s fmt of %s" % (m.group(1), m.group(3), m.group(2)) if m else "?"


models = {"installed": "checkpoints_lprnet/lprnet_avg.pt", "weather only": "checkpoints_compare_weather/fixed_s0/lprnet_avg.pt"}
tot = {}
print("exact plates on the held-out crops, three ways of reading: plain / smoothed / chosen")
print("%-5s %6s | %-14s | %-14s | %-14s | %-14s | %s" % ("fold", "crops", "installed", "weather only", "foggy-fold prev", "NEW session", "NEW model on heldout_rj250"))
for k in range(4):
    test = "%s/foggy_ft/verified_fold%d_test" % (D, k)
    prev = "checkpoints_foggy_ft/verified_fold%d/fixed_s0/lprnet_avg.pt" % k
    new = "checkpoints_session_ft/session_fold%d/fixed_s0/lprnet_avg.pt" % k
    if not os.path.exists(new):
        print("fold %d: no new model yet" % k)
        continue
    res = {n: score(net_of(c), test) for n, c in list(models.items()) + [("foggy-fold prev", prev), ("NEW session", new)]}
    for n, v in res.items():
        t = tot.setdefault(n, [0, 0, 0, 0])
        for i in range(4):
            t[i] += v[i]
    f = lambda v: "%2d / %2d / %2d" % v[:3]
    print("%-5d %6d | %-14s | %-14s | %-14s | %-14s | %s" % (k, res["installed"][3], f(res["installed"]), f(res["weather only"]), f(res["foggy-fold prev"]), f(res["NEW session"]), clear(new)))
print("TOTAL %6d | %s" % (tot["installed"][3], " | ".join("%s: %d / %d / %d" % (n, v[0], v[1], v[2]) for n, v in tot.items())))
print("(installed reader on heldout_rj250: 53 exact / 56 fmt of 62)")
