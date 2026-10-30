#!/bin/bash
# Held-out GoPro frames (video 7, never trained on): dehaze with the model before and after the GoPro training, then the sheet and the numbers.
source ~/mamba-venv/bin/activate
unset PYTORCH_CUDA_ALLOC_CONF
cd /mnt/c/projects/jewwww/uvm
G=../datasets/gopro_hazy
rm -rf $G/out_before $G/out_after; mkdir -p $G/out_before $G/out_after
python test_uvmnet.py --input $G/heldout_hazy --checkpoint checkpoints_ponmudi_gf/uvmnet_combined_final.pt --output $G/out_before --tile_size 1536 > logs/gp_before.log 2>&1
python test_uvmnet.py --input $G/heldout_hazy --checkpoint checkpoints_ponmudi_gopro/uvmnet_combined_final.pt --output $G/out_after --tile_size 1536 > logs/gp_after.log 2>&1
echo "restored: $(ls $G/out_before | wc -l) before, $(ls $G/out_after | wc -l) after"
python foggy_ft/gopro/eval_heldout.py 2>&1 | grep -v Warning
echo HELDOUT_FINISHED
