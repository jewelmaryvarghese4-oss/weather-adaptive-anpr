#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7
rm -rf checkpoints_real_smoke
python real_scratch/train_uvm_real.py --root ../datasets/real_scratch --out_dir checkpoints_real_smoke \
  --init_ckpt checkpoints_real_mixed/keep_epoch22_ema.pt --nhhaze_root ../datasets/1001hazeimages/NH-HAZE \
  --aligned_synth_samples 3000 --aligned_gopro_synth_frac 0.7 --aligned_nh_frac 0.5 --aligned_batch 3 --aligned_lambda 4.0 \
  --synth_beta_hi 6.0 --real_loss_scale 0.3 --lr 3e-5 --epochs 1 --samples 80 --batch 4 --gpu_fraction 0.9 --seed 1 2>&1 | tr '\r' '\n' | grep -a -v "^epoch 1/1" | tail -14
ls checkpoints_real_smoke
echo SMOKE_DONE
