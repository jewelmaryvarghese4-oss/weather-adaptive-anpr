"""
4-fold, split-by-plate fine-tuning data from the verified foggy crops.

    python foggy_ft/make_folds.py                      # uses datasets/foggy_ft/foggy_labels_verified.csv
    python foggy_ft/make_folds.py --labels X.csv --tag dryrun

Writes (all NEW files, nothing existing is touched):
  datasets/foggy_ft_manifest_<tag>_fold{k}.csv   training manifest of fold k = the rj250 manifest + the foggy crops of the other plates
  datasets/foggy_ft/<tag>_fold{k}_test/          crops/ + labels.json of the plates held out in fold k (for label_review.py eval)
  datasets/foggy_ft/<tag>_folds.json             which plate is in which fold

Every plate (identical text) goes to exactly one fold, so a plate is never in training and testing together. In training each plate
contributes at most --cap crops, each repeated --repeat times, so the three most common cars do not dominate.
"""
import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.abspath(os.path.join(HERE, "..", "..", "datasets"))
FT = os.path.join(D, "foggy_ft")

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default=os.path.join(FT, "foggy_labels_verified.csv"))
ap.add_argument("--base_manifest", default=os.path.join(D, "final_manifest_rj250_renamed.csv"))
ap.add_argument("--folds", type=int, default=4)
ap.add_argument("--cap", type=int, default=4)
ap.add_argument("--repeat", type=int, default=4)
ap.add_argument("--tag", default="verified")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rows = [r for r in csv.DictReader(open(a.labels, encoding="utf-8")) if r["status"] in ("ok", "fix") and r["label"]]
by_plate = defaultdict(list)
for r in rows:
    by_plate[r["label"].upper()].append(r)
print("verified crops: %d | distinct plates: %d" % (len(rows), len(by_plate)))

# balance the folds by number of crops: biggest plates first, each to the fold with the fewest crops so far
rng = random.Random(a.seed)
plates = sorted(by_plate, key=lambda p: (-len(by_plate[p]), rng.random()))
fold_of, load = {}, [0] * a.folds
for p in plates:
    k = min(range(a.folds), key=lambda i: (load[i], i))
    fold_of[p] = k
    load[k] += len(by_plate[p])
print("crops per fold:", load, "| plates per fold:", [sum(v == k for v in fold_of.values()) for k in range(a.folds)])

base = list(csv.reader(open(a.base_manifest, encoding="utf-8")))
hdr, body = base[0], base[1:]
for k in range(a.folds):
    test_dir = os.path.join(FT, "%s_fold%d_test" % (a.tag, k))
    shutil.rmtree(test_dir, ignore_errors=True)
    os.makedirs(os.path.join(test_dir, "crops"))
    labels, train = {}, []
    for p, rs in by_plate.items():
        if fold_of[p] == k:
            for r in rs:
                pid = "fog%03d" % int(r["number"])
                shutil.copy(os.path.join(FT, "crops", "%03d.png" % int(r["number"])), os.path.join(test_dir, "crops", pid + ".png"))
                labels[pid] = p
        else:
            keep = sorted(rs, key=lambda r: int(r["number"]))[:a.cap]
            train += [(("foggy_ft/crops/%03d.png" % int(r["number"])), p) for r in keep]
    json.dump(labels, open(os.path.join(test_dir, "labels.json"), "w"), indent=1)
    out = os.path.join(D, "foggy_ft_manifest_%s_fold%d.csv" % (a.tag, k))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(body)
        for path, text in train:
            for _ in range(a.repeat):
                w.writerow([path, text, "1", "foggy_train"] + [""] * (len(hdr) - 4))
    print("fold %d: test %d crops / %d plates | train foggy rows %d (x%d) | manifest -> %s" % (
        k, len(labels), len(set(labels.values())), len(train), a.repeat, os.path.basename(out)))
json.dump({"fold_of": fold_of, "crops_per_plate": {p: len(v) for p, v in by_plate.items()}}, open(os.path.join(FT, "%s_folds.json" % a.tag), "w"), indent=1)
