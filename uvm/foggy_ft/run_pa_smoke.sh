#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
R=../datasets/1001hazeimages
python train_uvm_plateaware.py --ots_root $R/OTS_BETA --plates_root $R/FoggyPlates --nhhaze_root $R/NH-HAZE \
  --synth_clear_root ../datasets/rainfog_clear $R/FoggyPlates/clear --synth_samples 16 --ponmudi_samples 16 \
  --ots_limit 40 --nhhaze_repeat 1 --lowlight_prob 0.15 --lr 1e-5 --epochs 1 --ckpt_every 1 --gf_radius 4 --gf_eps 0.01 \
  --plate_lambda 0.03 --plate_batch 8 \
  --init_ckpt checkpoints_ponmudi_gf/uvmnet_combined_final.pt --ckpt_dir checkpoints_pa_smoke > logs/pa_smoke.log 2>&1
echo "exit $?"
tr '\r' '\n' < logs/pa_smoke.log | grep -a -E "plate-aware|Error|Traceback|epoch +1 \|" | tail -8 | cut -c1-220
