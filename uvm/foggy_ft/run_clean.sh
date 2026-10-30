#!/bin/bash
# Seamless full-size dehazing + tint fix on foggy_photos, then the checks and the plate re-score. New outputs only.
source ~/mamba-venv/bin/activate
unset PYTORCH_CUDA_ALLOC_CONF
cd /mnt/c/projects/jewwww/uvm
D=../datasets
echo "start $(date +%T)"
python foggy_ft/clean/dehaze_clean.py --input $D/foggy_photos_upright --output $D/foggy_photos_uvm_gopro_clean 2>&1 | grep -v -E "^done|Warning|Input size|Processing|Output size|^$"
echo "dehazed: $(ls $D/foggy_photos_uvm_gopro_clean/*_restored.png 2>/dev/null | wc -l) photos at $(date +%T)"
python foggy_ft/clean/clean_check.py 2>&1 | grep -v Warning
python foggy_ft/clean/tint_plates.py 2>&1 | grep -v Warning
cd ~/dehaze-eval; export PYTHONPATH=~/dehaze-eval/extra
~/mamba-venv/bin/python /mnt/c/projects/jewwww/uvm/foggy_ft/shootout/shoot_score_select2.py raw uvm_gp+gf uvm_gpt 2>&1 | grep -v Warning
echo CLEAN_RUN_FINISHED
