#!/bin/bash
# Final long-run weights (epoch 20 = best heavy-fog epoch) on the 60 Ponmudi photos -> separate folder.
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7
python real_scratch/dehaze_photos.py --checkpoint checkpoints_real_long/best_heavy_ema.pt \
  --input ../datasets/foggy_photos_upright --output ../datasets/foggy_photos_uvm_real_long 2>&1 | tr '\r' '\n' | tail -2
echo REAL_LONG_DONE
