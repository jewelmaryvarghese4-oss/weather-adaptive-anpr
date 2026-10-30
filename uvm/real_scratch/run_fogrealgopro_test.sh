#!/bin/bash
# TEST ONLY (separate folder): same GoPro hybrid + tint fix pipeline on 5 images enlarged x3. Does not touch finalout/.
source ~/mamba-venv/bin/activate
unset PYTORCH_CUDA_ALLOC_CONF
cd /mnt/c/projects/jewwww/uvm
T=/mnt/c/projects/jewwww/datasets/finalrun_test_x3
IN=$T/in; W=$T/work; OUT=$T/out
mkdir -p $W $OUT
python test_uvmnet.py --input $IN --checkpoint checkpoints_ponmudi_gopro/uvmnet_combined_final.pt --output $W/raw --tile_size 1536 > $W/raw.log 2>&1
python foggy_ft/newfolder/nf_gf.py $IN $W/raw $W/gf | tail -1
python foggy_ft/clean/dehaze_clean.py --input $IN --output $W/clean_notint --tint_beta 0 2>&1 | grep -c "^done"
python foggy_ft/clean/hybrid.py --tint 0.8 --output $W/hybrid_tint --hazy_dir $IN --tiled_dir $W/gf --seam_dir $W/clean_notint
for f in $W/hybrid_tint/*_restored.png; do b=$(basename "$f" _restored.png); cp "$f" "$OUT/$b.png"; done
echo "test outputs: $(ls $OUT/*.png | wc -l)"
echo TEST_X3_DONE
