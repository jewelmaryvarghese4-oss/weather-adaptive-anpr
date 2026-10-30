#!/bin/bash
# Plate-aware UVM: starts from the filter-trained checkpoint, adds the frozen-reader CTC loss on real hazy plate windows.
# New outputs only: checkpoints_ponmudi_plate/, logs/pa_*, ~/dehaze-eval/out/uvm_pa.
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
python foggy_ft/patch_shoot_pa.py
R=../datasets/1001hazeimages
echo "training starts $(date +%T)"
python train_uvm_plateaware.py --ots_root $R/OTS_BETA --plates_root $R/FoggyPlates --nhhaze_root $R/NH-HAZE \
  --synth_clear_root ../datasets/rainfog_clear $R/FoggyPlates/clear --synth_samples 1000 --ponmudi_samples 2000 \
  --ots_limit 1000 --nhhaze_repeat 10 --lowlight_prob 0.15 --lr 1e-5 --epochs 5 --ckpt_every 5 --gf_radius 4 --gf_eps 0.01 \
  --plate_lambda 0.03 --plate_batch 8 \
  --init_ckpt checkpoints_ponmudi_gf/uvmnet_combined_final.pt --ckpt_dir checkpoints_ponmudi_plate > logs/pa_uvm_ft.log 2>&1
tr '\r' '\n' < logs/pa_uvm_ft.log | grep -a -E "plate-aware|^epoch +[0-9]+ \|" | cut -c1-200
[ -f checkpoints_ponmudi_plate/uvmnet_combined_final.pt ] || { echo "PA_TRAINING_FAILED"; exit 1; }
echo "training finished $(date +%T); dehazing the 50 photos at 1536 px"
cd ~/dehaze-eval
export PYTHONPATH=~/dehaze-eval/extra
~/mamba-venv/bin/python scripts/shoot_dehaze.py --model uvm_pa > logs/uvm_pa.log 2>&1
tail -1 logs/uvm_pa.log
~/mamba-venv/bin/python /mnt/c/projects/jewwww/uvm/foggy_ft/shootout/shoot_score_select2.py raw uvm_gf+gf uvm_pa uvm_pa+gf 2>&1 | grep -v Warning
echo PA_RUN_FINISHED
