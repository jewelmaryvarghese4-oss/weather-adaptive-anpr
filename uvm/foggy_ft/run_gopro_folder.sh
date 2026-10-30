#!/bin/bash
# GoPro-trained UVM on the whole foggy_photos folder (upright copies), full size, then the fog check. New outputs only.
source ~/mamba-venv/bin/activate
unset PYTORCH_CUDA_ALLOC_CONF
cd /mnt/c/projects/jewwww/uvm
D=../datasets
while ps -eo args | grep -E "test_uvmnet|run_gopro_heldout|train_uvm" | grep -vq grep; do sleep 15; done
mkdir -p $D/foggy_photos_uvm_gopro
echo "start $(date +%T)"
python test_uvmnet.py --input $D/foggy_photos_upright --checkpoint checkpoints_ponmudi_gopro/uvmnet_combined_final.pt --output $D/foggy_photos_uvm_gopro --tile_size 1536 > $D/foggy_photos_uvm_gopro/dehaze_log.txt 2>&1
echo "dehazed: $(ls $D/foggy_photos_uvm_gopro/*_restored.png 2>/dev/null | wc -l) photos at $(date +%T)"
python foggy_ft/gopro_folder_check.py 2>&1 | grep -v -E "^done|Warning"
echo FOLDER_CHECK_FINISHED
