#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7
rm -rf checkpoints_depthfog_smoke
python real_scratch/train_uvm_depthfog.py --root ../datasets/real_scratch --out_dir checkpoints_depthfog_smoke \
  --init_ckpt checkpoints_real_long/best_heavy_ema.pt --nhhaze_root ../datasets/1001hazeimages/NH-HAZE \
  --aligned_synth_samples 3000 --depthfog_root ../datasets/depthfog --aligned_depth_frac 0.8 --aligned_nh_frac 0.5 \
  --aligned_batch 3 --aligned_lambda 4.0 --real_loss_scale 0.3 --lr 5e-5 --epochs 1 --samples 80 --batch 4 --gpu_fraction 0.9 --seed 2 2>&1 | tr '\r' '\n' | grep -a -v "^epoch 1/1" | grep -a -v -i warn | tail -16
ls checkpoints_depthfog_smoke
echo SMOKE2_DONE
