#!/bin/bash
# GoPro hybrid + tint fix (same steps as foggy_ft/newfolder/nf_fullsize.sh, strength 0.8) on finalin/foggysynthetic -> finalout/fogrealgopro/<name>.png
source ~/mamba-venv/bin/activate
unset PYTORCH_CUDA_ALLOC_CONF
cd /mnt/c/projects/jewwww/uvm
IN=/mnt/c/projects/jewwww/finalin/foggysynthetic
W=/mnt/c/projects/jewwww/datasets/finalrun_foggysynthetic
OUT=/mnt/c/projects/jewwww/finalout/fogrealgopro
mkdir -p $W $OUT
echo "1/4 GoPro model (tile 1536)"
python test_uvmnet.py --input $IN --checkpoint checkpoints_ponmudi_gopro/uvmnet_combined_final.pt --output $W/raw --tile_size 1536 > $W/raw.log 2>&1
echo "raw: $(ls $W/raw/*_restored.png 2>/dev/null | wc -l) images"
echo "2/4 guided filter r4"
python foggy_ft/newfolder/nf_gf.py $IN $W/raw $W/gf | tail -1
echo "3/4 seam-free single pass (no tint)"
python foggy_ft/clean/dehaze_clean.py --input $IN --output $W/clean_notint --tint_beta 0 2>&1 | grep -c "^done"
echo "4/4 hybrid + tint fix 0.8"
python foggy_ft/clean/hybrid.py --tint 0.8 --output $W/hybrid_tint --hazy_dir $IN --tiled_dir $W/gf --seam_dir $W/clean_notint
echo "copy to output, names like the input"
for f in $W/hybrid_tint/*_restored.png; do b=$(basename "$f" _restored.png); cp "$f" "$OUT/$b.png"; done
echo "output files: $(ls $OUT/*.png | wc -l)"
echo FOGREALGOPRO_DONE
