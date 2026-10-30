#!/bin/bash
# Dehaze the 4 new foggy photos with several kept epochs so they can be compared side by side.
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7
for n in 19 22 23 24; do
  python real_scratch/dehaze_photos.py --checkpoint checkpoints_real_mixed/keep_epoch${n}_ema.pt \
    --input ../datasets/newfoggyphotos/results/upright --output ../datasets/newfoggyphotos/results/real_ep${n} 2>&1 | tr '\r' '\n' | tail -2
done
echo COMPARE_DONE
