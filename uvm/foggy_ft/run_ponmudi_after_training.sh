#!/bin/bash
# Waits for the reader 4-fold training to finish, then trains UVM with the clear Ponmudi photos + calibrated fog, dehazes the 50 test photos
# with it and scores plate reading on the 85 verified crops. Writes only new things (checkpoints_ponmudi/, logs/, ~/dehaze-eval/out/uvm_ponmudi).
cd /mnt/c/projects/jewwww/uvm
LOG=logs/session_ft_all.log
idle=0
until grep -aq "TOTAL" $LOG 2>/dev/null; do
  if ps -eo args | grep -E "train_lprnet|eval_folds_session|compare_two_line" | grep -vq grep; then idle=0; else idle=$((idle+1)); fi
  if [ $idle -ge 5 ]; then echo "WAITING_ABORTED: the reader training is no longer running and never printed its table"; exit 1; fi
  sleep 60
done
echo "reader training finished at $(date +%T); starting the Ponmudi UVM training"
source ~/mamba-venv/bin/activate
R=../datasets/1001hazeimages
python train_uvm_ponmudi.py --ots_root $R/OTS_BETA --plates_root $R/FoggyPlates --nhhaze_root $R/NH-HAZE \
  --synth_clear_root ../datasets/rainfog_clear $R/FoggyPlates/clear --synth_samples 1500 --ponmudi_samples 3000 \
  --ots_limit 2000 --nhhaze_repeat 10 --lowlight_prob 0.15 --lr 1.5e-5 --epochs 6 --ckpt_every 2 \
  --init_ckpt checkpoints_rainfog2/uvmnet_combined_final.pt --ckpt_dir checkpoints_ponmudi > logs/ponmudi_uvm_ft.log 2>&1
tail -3 logs/ponmudi_uvm_ft.log | cut -c1-200
[ -f checkpoints_ponmudi/uvmnet_combined_final.pt ] || { echo "UVM_TRAINING_FAILED"; exit 1; }
echo "UVM training finished at $(date +%T); dehazing the 50 photos"
cd ~/dehaze-eval
export PYTHONPATH=~/dehaze-eval/extra
~/mamba-venv/bin/python scripts/shoot_dehaze.py --model uvm_ponmudi > logs/uvm_ponmudi.log 2>&1
tail -1 logs/uvm_ponmudi.log
~/mamba-venv/bin/python scripts/shoot_score.py raw uvm_ft2 uvm_ft2+gf uvm_ponmudi uvm_ponmudi+gf 2>&1 | grep -v Warning
~/mamba-venv/bin/python /mnt/c/projects/jewwww/uvm/foggy_ft/ponmudi_eval.py
echo PONMUDI_STEP_FINISHED
