#!/bin/bash
# Small test: does a mild denoise help on the noisy rainy photos? Compares plate reading on 6 photos in 5 versions.
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
D=/mnt/c/projects/jewwww/datasets
O=$D/foggy_denoise_test
CK=checkpoints_combined/uvmnet_combined_best_stateonly.pt
PHOTOS="20260926_111504 20260926_111507 20260926_111508 20260926_111526 20260926_111532 20260926_111604"
rm -rf $O; mkdir -p $O/v1_raw $O/in_h3 $O/in_h5 $O/v2_uvm $O/v3_den3_uvm $O/v4_den5_uvm $O/v5_uvm_den5
for p in $PHOTOS; do
  cp "$D/foggy_photos/$p.jpg" $O/v1_raw/
  cp "$D/foggy_photos_dehazed/${p}_restored.png" $O/v2_uvm/${p}.png
done
python - <<EOF
import cv2, glob, os
O = "$O"
for p in "$PHOTOS".split():
    im = cv2.imread("$D/foggy_photos/%s.jpg" % p)
    cv2.imwrite("%s/in_h3/%s.png" % (O, p), cv2.fastNlMeansDenoisingColored(im, None, 3, 3, 7, 21))
    cv2.imwrite("%s/in_h5/%s.png" % (O, p), cv2.fastNlMeansDenoisingColored(im, None, 5, 5, 7, 21))
    d = cv2.imread("%s/v2_uvm/%s.png" % (O, p))
    cv2.imwrite("%s/v5_uvm_den5/%s.png" % (O, p), cv2.fastNlMeansDenoisingColored(d, None, 5, 5, 7, 21))
print("denoised inputs written", flush=True)
EOF
python test_uvmnet.py --input $O/in_h3 --checkpoint $CK --output $O/v3_den3_uvm --tile_size 1536 > $O/log_h3.txt 2>&1
python test_uvmnet.py --input $O/in_h5 --checkpoint $CK --output $O/v4_den5_uvm --tile_size 1536 > $O/log_h5.txt 2>&1
for v in v1_raw v2_uvm v3_den3_uvm v4_den5_uvm v5_uvm_den5; do
  python read_plate.py --image_dir $O/$v --skip_dehaze --save_annotated $O/$v/annotated > $O/read_$v.txt 2>&1
done
python - <<EOF
import re
O = "$O"
truth = {"KL35N8421", "KL80A5787", "KL25N5018", "KL74B8479", "KL16P6373"}   # plates read by eye on the clear photos
rows = []
for v in "v1_raw v2_uvm v3_den3_uvm v4_den5_uvm v5_uvm_den5".split():
    txt = open("%s/read_%s.txt" % (O, v), errors="ignore").read()
    reads = []
    for m in re.finditer(r"plate \d+\s+box=.*\n\s+raw image\s+-> predicted: '([^']*)'(?: -> format-corrected: '([^']*)')?", txt):
        reads.append(m.group(2) or m.group(1))
    hits = [r for r in reads if r in truth]
    rows.append((v, len(reads), len(hits), sorted(set(hits))))
with open(O + "/summary.txt", "w") as f:
    for v, n, h, s in rows:
        line = "%-14s plates detected %2d | exact matches with the known plates %2d %s" % (v, n, h, s)
        print(line); f.write(line + "\n")
EOF
echo FINISHED
