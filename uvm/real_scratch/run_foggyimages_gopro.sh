#!/bin/bash
# GoPro hybrid + tint fix (strength 0.8; same steps as foggy_ft/newfolder/nf_fullsize.sh) on finalin/foggyimages -> finalout/fogrealgopro/<name>.png
# Works from upright (EXIF-rotated) PNG copies made in the work folder; the input folder is not touched.
source ~/mamba-venv/bin/activate
unset PYTORCH_CUDA_ALLOC_CONF
cd /mnt/c/projects/jewwww/uvm
SRC=/mnt/c/projects/jewwww/finalin/foggyimages
W=/mnt/c/projects/jewwww/datasets/finalrun_foggyimages
IN=$W/upright
OUT=/mnt/c/projects/jewwww/finalout/fogrealgopro
rm -rf $W/raw $W/gf $W/clean_notint $W/hybrid_tint $IN
mkdir -p $W $OUT $IN
echo "0/4 upright copies"
python - <<EOF
import glob, os
from PIL import Image, ImageOps
n = 0
for f in sorted(glob.glob("$SRC/*.jpg") + glob.glob("$SRC/*.JPG") + glob.glob("$SRC/*.png")):
    im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    im.save("$IN/" + os.path.splitext(os.path.basename(f))[0] + ".png", compress_level=3)
    n += 1
print("upright copies:", n)
EOF
echo "1/4 GoPro model (tile 1536)"
python test_uvmnet.py --input $IN --checkpoint checkpoints_ponmudi_gopro/uvmnet_combined_final.pt --output $W/raw --tile_size 1536 > $W/raw.log 2>&1
echo "raw: $(ls $W/raw/*_restored.png 2>/dev/null | wc -l) images"
echo "2/4 guided filter r4"
python foggy_ft/newfolder/nf_gf.py $IN $W/raw $W/gf 2>&1 | tail -1
echo "gf: $(ls $W/gf/*_restored.png 2>/dev/null | wc -l) images"
echo "3/4 seam-free single pass (no tint)"
python foggy_ft/clean/dehaze_clean.py --input $IN --output $W/clean_notint --tint_beta 0 2>&1 | grep -c "^done"
echo "4/4 hybrid + tint fix 0.8"
python foggy_ft/clean/hybrid.py --tint 0.8 --output $W/hybrid_tint --hazy_dir $IN --tiled_dir $W/gf --seam_dir $W/clean_notint
echo "copy to output, names like the input"
for f in $W/hybrid_tint/*_restored.png; do b=$(basename "$f" _restored.png); cp "$f" "$OUT/$b.png"; done
echo "output files: $(ls $OUT/*.png | wc -l)"
echo FOGGYIMAGES_DONE
