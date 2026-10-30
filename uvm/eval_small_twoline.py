"""
Proxy test for "can the model read small two-line plates (bike / auto plates in video)?".

Takes every two-line crop with known text from the labeled sets, shrinks each one to bike-plate
size with degrade.degrade_crop() using a FIXED random seed (so every checkpoint sees exactly the
same degraded images), and reports how often each checkpoint reads them exactly.

Note: crops that a checkpoint trained on are 'seen' in clean form, so use this to COMPARE
checkpoints trained on the same data, not as an absolute accuracy.

    python eval_small_twoline.py ckpt1.pt ckpt2.pt ...
"""
import glob
import json
import os
import random
import sys

import torch
from PIL import Image

from degrade import degrade_crop
from model_lprnet import LPRNet
from read_plate import read_plate
from two_line_plate import is_two_line

D = "../datasets"
VARIANTS = 6   # degraded copies per plate


def labeled_two_line():
    items = []
    for sub in ("heldout_test", "heldout_test_ds11"):
        labels = json.load(open(f"{D}/{sub}/labels.json"))
        for k, t in labels.items():
            p = glob.glob(f"{D}/{sub}/crops/{k}.*")
            if p:
                items.append((k, Image.open(p[0]).convert("RGB"), t))
    return [(k, im, t) for k, im, t in items if is_two_line(im)]


def main():
    ckpts = sys.argv[1:]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = labeled_two_line()
    random.seed(1234)
    samples = []                                    # (truth, degraded image), same for every checkpoint
    for k, im, t in items:
        for _ in range(VARIANTS):
            samples.append((t, degrade_crop(im, min_h=18, max_h=40)))
    print(f"{len(items)} two-line plates x {VARIANTS} degraded copies = {len(samples)} test images")
    for ck in ckpts:
        net = LPRNet(num_classes=37).to(dev)
        net.load_state_dict(torch.load(ck, map_location=dev))
        net.eval()
        reads = [read_plate(net, im, dev)[1] for _, im in samples]
        exact = sum(r == t for r, (t, _) in zip(reads, samples))
        from train_lprnet import levenshtein
        cer = sum(levenshtein(r, t) / len(t) for r, (t, _) in zip(reads, samples)) / len(samples)
        print(f"{ck:<60} exact {exact:>3}/{len(samples)} = {exact / len(samples):5.1%}   CER {cer:.3f}")


if __name__ == "__main__":
    main()
