#!/bin/bash
# Parallel evaluation home for the new photo folder: same repos/weights/code as ~/dehaze-eval, but its own data/ and out/ (inside the results folder).
# Run every dehazer script with  HOME=/home/jewel/nfhome2  so that "~/dehaze-eval" points here; nothing of the earlier results is touched.
set -e
R=/mnt/c/projects/jewwww/datasets/finalrun_foggyimages_gallery/results
NF=/home/jewel/nfhome2
mkdir -p $R/in1536 $R/s1536 $R/upright $R/full $NF/dehaze-eval/data $NF/dehaze-eval/run $NF/dehaze-eval/logs
for d in repos weights extra shim scripts; do ln -sfn /home/jewel/dehaze-eval/$d $NF/dehaze-eval/$d; done
ln -sfn $R/in1536 $NF/dehaze-eval/data/in1536
ln -sfn $R/s1536 $NF/dehaze-eval/out
ln -sfn /home/jewel/mamba-venv $NF/mamba-venv
ln -sfn /home/jewel/.cache $NF/.cache
# upright copies + 1536 px inputs
/home/jewel/mamba-venv/bin/python - <<'PY'
import glob, os
from PIL import Image, ImageOps
R = "/mnt/c/projects/jewwww/datasets/finalrun_foggyimages_gallery/results"
for f in sorted(glob.glob("/mnt/c/projects/jewwww/finalin/foggyimages/*.jpg")):
    stem = os.path.splitext(os.path.basename(f))[0]
    im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    im.save("%s/upright/%s.jpg" % (R, stem), quality=95, subsampling=0)
    s = min(1.0, 1536 / max(im.size))
    im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS).save("%s/in1536/%s.png" % (R, stem))
    print(stem, im.size, "->", round(im.width * s), round(im.height * s))
PY
echo "setup done: $(ls $R/in1536 | wc -l) photos"
