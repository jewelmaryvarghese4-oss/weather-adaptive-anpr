#!/bin/bash
source ~/mamba-venv/bin/activate
unset PYTORCH_CUDA_ALLOC_CONF
cd /mnt/c/projects/jewwww/uvm
D=../datasets
echo "start $(date +%T)"
python foggy_ft/clean/dehaze_clean.py --input $D/foggy_photos_upright --output $D/foggy_photos_uvm_gopro_clean_notint --tint_beta 0 2>&1 | grep -v -E "^done|Warning|Input size|Processing|Output size|^$"
echo "dehazed: $(ls $D/foggy_photos_uvm_gopro_clean_notint/*_restored.png | wc -l) photos at $(date +%T)"
python foggy_ft/clean/make_tint_sheet.py
echo NOTINT_FINISHED
