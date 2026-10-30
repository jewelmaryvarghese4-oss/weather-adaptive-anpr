#!/bin/bash
# Ponmudi UVM on the whole foggy_photos folder (upright copies), full size, then the fog check. New outputs only.
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
D=../datasets
[ -d $D/foggy_photos_upright ] || { echo "no upright copy"; exit 1; }
mkdir -p $D/foggy_photos_uvm_ponmudi
echo "start $(date +%T)"
python test_uvmnet.py --input $D/foggy_photos_upright --checkpoint checkpoints_ponmudi/uvmnet_combined_final.pt --output $D/foggy_photos_uvm_ponmudi --tile_size 1536 > $D/foggy_photos_uvm_ponmudi/dehaze_log.txt 2>&1
echo "dehazed: $(ls $D/foggy_photos_uvm_ponmudi/*_restored.png 2>/dev/null | wc -l) photos at $(date +%T)"
python foggy_ft/ponmudi_folder_check.py
echo FOLDER_CHECK_FINISHED
