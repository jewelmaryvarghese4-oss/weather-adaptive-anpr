#!/bin/bash
# Full-size UVM family on the new photos -> results/full/<name>/<stem>_restored.png (same variants as the gallery of the first folder)
source ~/mamba-venv/bin/activate
unset PYTORCH_CUDA_ALLOC_CONF
cd /mnt/c/projects/jewwww/uvm
R=/mnt/c/projects/jewwww/datasets/newfoggyphotos/results
UP=$R/upright; F=$R/full
mkdir -p $F/ft2 $F/pon $F/_raw_gf $F/_raw_gopro
for cfg in "ft2 checkpoints_rainfog2" "pon checkpoints_ponmudi" "_raw_gf checkpoints_ponmudi_gf" "_raw_gopro checkpoints_ponmudi_gopro"; do
  set -- $cfg
  python test_uvmnet.py --input $UP --checkpoint $2/uvmnet_combined_final.pt --output $F/$1 --tile_size 1536 > $F/$1.log 2>&1
  echo "$1: $(ls $F/$1/*_restored.png 2>/dev/null | wc -l) photos"
done
python foggy_ft/newfolder/nf_gf.py $UP $F/_raw_gf $F/gf | tail -1
python foggy_ft/newfolder/nf_gf.py $UP $F/_raw_gopro $F/gopro_gf | tail -1
python foggy_ft/clean/dehaze_clean.py --input $UP --output $F/gopro_clean --tint_beta 0.8 2>&1 | grep -c "^done"
python foggy_ft/clean/dehaze_clean.py --input $UP --output $F/gopro_clean_notint --tint_beta 0 2>&1 | grep -c "^done"
python foggy_ft/clean/hybrid.py --tint 0.0 --output $F/hybrid --hazy_dir $UP --tiled_dir $F/gopro_gf --seam_dir $F/gopro_clean_notint
python foggy_ft/clean/hybrid.py --tint 0.8 --output $F/hybrid_tint --hazy_dir $UP --tiled_dir $F/gopro_gf --seam_dir $F/gopro_clean_notint
echo NF_FULL_FINISHED
