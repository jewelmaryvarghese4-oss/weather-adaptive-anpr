#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7
python real_scratch/dehaze_photos.py --checkpoint checkpoints_real_mixed/keep_epoch22_ema.pt \
  --input ../datasets/foggy_photos_upright --output ../datasets/foggy_photos_uvm_real 2>&1 | tr '\r' '\n' | tail -2
echo REAL60_DONE
