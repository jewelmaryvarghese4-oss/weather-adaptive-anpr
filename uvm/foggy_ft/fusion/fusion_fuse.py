"""
Option 4, step 2 (SEPARATE from every earlier result): fuse the crops of one tracked plate into a single cleaner image, and read it.
For every track in datasets/fusion/tracks/: keep the sharpest/widest crops -> upsample x2 -> align each to the sharpest one (ECC, affine) ->
weighted mean (weights = sharpness x alignment score) = "fused", and median of the best 7 = "fused_med".
Reads, for the installed reader and the 3 weather-trained readers (plain-or-smoothed choice, as in the earlier plate scoring):
  single    the widest crop alone
  vote      most common valid reading over all crops of the track (what the earlier tracking did)
  fused / fused_med   the fused image
Ground truth: a track counts as 'matched' when the installed reader's vote over all crops is within 1 edit of exactly one verified foggy plate (same rule as
build_video_tracks.py); the verified plate is then the truth.  (This favours tracks that are already nearly readable, so gains are, if anything, understated.)
    python foggy_ft/fusion/fusion_fuse.py          (run in WSL from uvm/)
"""
import argparse
import csv
import glob
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
UVM = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(UVM)
D = os.path.join(ROOT, "datasets")
os.chdir(UVM)
sys.path.insert(0, UVM)
os.environ["TWO_LINE_MODE"] = "fixed"

import cv2
import numpy as np
import torch
from PIL import Image

import read_plate as rp
from model_lprnet import LPRNet

ap = argparse.ArgumentParser()
ap.add_argument("--fusion", default=os.path.join(D, "fusion"))
ap.add_argument("--max_use", type=int, default=16, help="crops used per track")
ap.add_argument("--median_n", type=int, default=7)
ap.add_argument("--min_cc", type=float, default=0.55, help="ECC alignment score needed to use a crop")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = torch.device(a.device)
torch.set_grad_enabled(False)
READERS = [("installed", "checkpoints_lprnet/lprnet_avg.pt"), ("weather s0", "checkpoints_compare_weather/fixed_s0/lprnet_avg.pt"),
           ("weather s1", "checkpoints_compare_weather/fixed_s1/lprnet_avg.pt"), ("weather s2", "checkpoints_compare_weather/fixed_s2/lprnet_avg.pt")]


def lev(x, y):
    p = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        q = [i]
        for j, cy in enumerate(y, 1):
            q.append(min(p[j] + 1, q[j - 1] + 1, p[j - 1] + (cx != cy)))
        p = q
    return p[-1]


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


def chosen(net, im):
    ta, ca, va = read(net, im)
    tb, cb, vb = read(net, word(im))
    return max([(va, ca, ta), (vb, cb, tb)])        # (valid, conf, text)


def pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def sharp(g):
    return float(cv2.Laplacian(cv2.resize(g, (128, max(8, round(128 * g.shape[0] / g.shape[1])))), cv2.CV_32F).var())


def fuse(crops, S=2):
    """crops: list of BGR uint8. Returns (fused, fused_med, n_used)"""
    grays = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in crops]
    score = [sharp(g) * c.shape[1] for g, c in zip(grays, crops)]
    order = np.argsort(score)[::-1][:a.max_use]
    ref = crops[order[0]]
    H, W = ref.shape[0] * S, ref.shape[1] * S
    rimg = cv2.resize(ref, (W, H), interpolation=cv2.INTER_CUBIC)
    rg = cv2.GaussianBlur(cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    warped, weights = [rimg.astype(np.float32)], [score[order[0]]]
    for k in order[1:]:
        img = cv2.resize(crops[k], (W, H), interpolation=cv2.INTER_CUBIC)
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(rg, g, warp, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
        except cv2.error:
            continue
        if cc < a.min_cc:
            continue
        w = cv2.warpAffine(img, warp, (W, H), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        warped.append(w.astype(np.float32))
        weights.append(score[k] * cc)
    wts = np.array(weights, np.float32)
    mean = np.tensordot(wts / wts.sum(), np.stack(warped), axes=1)
    med = np.median(np.stack(warped[:a.median_n]), axis=0)
    return np.clip(mean + 0.5, 0, 255).astype(np.uint8), np.clip(med + 0.5, 0, 255).astype(np.uint8), len(warped)


def main():
    nets = []
    for name, ck in READERS:
        net = LPRNet(num_classes=37).to(dev)
        net.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
        net.eval()
        nets.append((name, net))
    ver = sorted({r["label"] for r in csv.DictReader(open(os.path.join(D, "foggy_ft", "foggy_labels_verified.csv"), encoding="utf-8")) if r["status"] in ("ok", "fix", "likely") and r["label"]})
    tdirs = sorted(glob.glob(os.path.join(a.fusion, "tracks", "*")))
    os.makedirs(os.path.join(a.fusion, "fused"), exist_ok=True)
    rows, sheet = [], []
    for td in tdirs:
        files = sorted(glob.glob(os.path.join(td, "*.png")))
        crops = [c for c in (cv2.imread(f) for f in files) if c is not None and c.shape[0] >= 8 and c.shape[1] >= 24]
        if len(crops) < 3:
            continue
        widest = crops[int(np.argmax([c.shape[1] for c in crops]))]
        fused, fmed, n_used = fuse(crops)
        tid = os.path.basename(td)
        cv2.imwrite(os.path.join(a.fusion, "fused", tid + "_fused.png"), fused)
        cv2.imwrite(os.path.join(a.fusion, "fused", tid + "_median.png"), fmed)
        # truth: the installed reader's vote over every crop
        net0 = nets[0][1]
        reads0 = [chosen(net0, pil(c)) for c in crops]
        valid0 = [t for v, cf, t in reads0 if v]
        cons = Counter(valid0).most_common(1)[0][0] if valid0 else ""
        truth = ""
        if cons:
            d = sorted((lev(cons, v), v) for v in ver)
            if d[0][0] <= 1 and (len(d) < 2 or d[1][0] > 1):
                truth = d[0][1]
        res = {}
        for name, net in nets:
            rd = [chosen(net, pil(c)) for c in crops]
            vv = [t for v, cf, t in rd if v]
            vote = Counter(vv).most_common(1)[0][0] if vv else ""
            res[name] = (chosen(net, pil(widest))[2], vote, chosen(net, pil(fused))[2], chosen(net, pil(fmed))[2])
        rows.append((tid, len(crops), n_used, cons, truth, res))
        if truth and len(sheet) < 16:
            sheet.append((truth, widest, fused, res["installed"][2], res["installed"][0]))
        print("%-14s crops %2d used %2d | vote(installed) %-11s truth %-11s | single %-11s fused %-11s" % (tid, len(crops), n_used, cons, truth or "-", res["installed"][0], res["installed"][2]), flush=True)
    m = [r for r in rows if r[4]]
    print("\ntracks: %d | matched to a verified foggy plate: %d" % (len(rows), len(m)))
    print("exact plates on the matched tracks (single / vote / fused / fused-median):")
    for name, _ in READERS:
        c = [sum(r[5][name][k] == r[4] for r in m) for k in range(4)]
        print("  %-12s %d / %d / %d / %d   of %d" % (name, *c, len(m)))
    with open(os.path.join(a.fusion, "fusion_results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["track", "crops", "used", "installed_vote_cons", "verified_truth"] + ["%s_%s" % (n, k) for n, _ in READERS for k in ("single", "vote", "fused", "median")])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4]] + [x for n, _ in READERS for x in r[5][n]])
    if sheet:
        H = 90
        out = []
        for truth, wid, fus, fread, sread in sheet:
            def tile(im):
                return cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H), interpolation=cv2.INTER_CUBIC)
            row = np.hstack([tile(wid), np.full((H, 8, 3), 255, np.uint8), tile(fus)])
            cv2.putText(row, "%s | single %s | fused %s" % (truth, sread or "-", fread or "-"), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
            out.append(row)
        wmax = max(r.shape[1] for r in out)
        out = [np.hstack([r, np.full((H, wmax - r.shape[1], 3), 255, np.uint8)]) for r in out]
        cv2.imwrite(os.path.join(a.fusion, "fusion_sheet.jpg"), np.vstack(out), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("sheet ->", os.path.join(a.fusion, "fusion_sheet.jpg"))
    print("FUSION_FUSE_DONE")


if __name__ == "__main__":
    main()
