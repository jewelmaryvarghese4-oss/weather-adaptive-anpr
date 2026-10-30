#!/bin/bash
# Dehazes 6 real rainy photos with a UVMNet checkpoint, reads the plates, and counts exact matches with the 5 plates read by eye
# on the clear photos. Usage:  bash eval_foggy_ckpt.sh <checkpoint.pt> <name>
# Results go to ../datasets/foggy_eval/<name>/  (summary.txt there). The untouched photos are the bar to beat (3 of 5 plates).
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
CK=${1:-checkpoints_combined/uvmnet_combined_best_stateonly.pt}
NAME=${2:-baseline}
D=/mnt/c/projects/jewwww/datasets
O=$D/foggy_eval/$NAME
PHOTOS="20260926_111504 20260926_111507 20260926_111508 20260926_111526 20260926_111532 20260926_111604"
rm -rf $O; mkdir -p $O/in $O/out
for p in $PHOTOS; do cp "$D/foggy_photos/$p.jpg" $O/in/; done
python test_uvmnet.py --input $O/in --checkpoint $CK --output $O/out --tile_size 1536 > $O/dehaze_log.txt 2>&1
python read_plate.py --image_dir $O/out --skip_dehaze --save_annotated $O/annotated > $O/read.txt 2>&1
python read_plate.py --image_dir $O/in --skip_dehaze > $O/read_raw.txt 2>&1
python - <<EOF
import re
import numpy as np
from PIL import Image
O = "$O"
truth = {"KL35N8421", "KL80A5787", "KL25N5018", "KL74B8479", "KL16P6373"}
def score(path):
    t = open(path, errors="ignore").read()
    reads = [m.group(2) or m.group(1) for m in re.finditer(r"raw image\s+-> predicted: '([^']*)'(?:\s+-> format-corrected: '([^']*)')?", t)]
    hits = sorted(set(r for r in reads if r in truth))
    return len(reads), hits, sum(r in truth for r in reads)
lines = []
for label, path in (("untouched photos", O + "/read_raw.txt"), ("dehazed with $NAME", O + "/read.txt")):
    n, hits, tot = score(path)
    lines.append("%-24s plates detected %2d | known plates found %d of 5 %s | exact hits %d" % (label, n, len(hits), hits, tot))
ph = "$PHOTOS".split()
b = []
for p in ph:
    a = np.asarray(Image.open("$D/foggy_photos/%s.jpg" % p).convert("L"), float).mean()
    c = np.asarray(Image.open("%s/out/%s_restored.png" % (O, p)).convert("L"), float).mean()
    b.append((a, c))
lines.append("mean brightness raw %.0f -> dehazed %.0f   (clear photos in the folder: about 152-158)" % (np.mean([x for x, _ in b]), np.mean([y for _, y in b])))
open(O + "/summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
