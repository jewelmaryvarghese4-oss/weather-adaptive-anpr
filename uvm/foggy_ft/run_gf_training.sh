#!/bin/bash
# UVM trained THROUGH a guided-filter layer, with Ponmudi patches at several photo scales, starting from the Ponmudi checkpoint.
# New outputs only: checkpoints_ponmudi_gf/, logs/gf_*, ~/dehaze-eval/out/uvm_gf, datasets/foggy_photos_uvm_gf.
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
python /mnt/c/Users/Jewel/AppData/Local/Temp/ppt/patch_shoot_gf.py
R=../datasets/1001hazeimages
echo "training starts $(date +%T)"
python train_uvm_gf.py --ots_root $R/OTS_BETA --plates_root $R/FoggyPlates --nhhaze_root $R/NH-HAZE \
  --synth_clear_root ../datasets/rainfog_clear $R/FoggyPlates/clear --synth_samples 1500 --ponmudi_samples 3000 \
  --ots_limit 2000 --nhhaze_repeat 10 --lowlight_prob 0.15 --lr 1.5e-5 --epochs 6 --ckpt_every 2 --gf_radius 4 --gf_eps 0.01 \
  --init_ckpt checkpoints_ponmudi/uvmnet_combined_final.pt --ckpt_dir checkpoints_ponmudi_gf > logs/gf_uvm_ft.log 2>&1
tail -3 logs/gf_uvm_ft.log | cut -c1-200
[ -f checkpoints_ponmudi_gf/uvmnet_combined_final.pt ] || { echo "GF_TRAINING_FAILED"; exit 1; }
echo "training finished $(date +%T); dehazing the 50 photos at 1536 px"
cd ~/dehaze-eval
export PYTHONPATH=~/dehaze-eval/extra
~/mamba-venv/bin/python scripts/shoot_dehaze.py --model uvm_gf > logs/uvm_gf.log 2>&1
tail -1 logs/uvm_gf.log
~/mamba-venv/bin/python /mnt/c/projects/jewwww/uvm/foggy_ft/shootout/shoot_score_select2.py raw uvm_ft2+gf uvm_ponmudi+gf uvm_gf uvm_gf+gf 2>&1 | grep -v Warning
~/mamba-venv/bin/python /mnt/c/projects/jewwww/uvm/foggy_ft/ponmudi_eval.py
echo "1536 px EVALUATION DONE"
cd /mnt/c/projects/jewwww/uvm
echo "full-size dehazing of the 60 photos $(date +%T)"
mkdir -p ../datasets/foggy_photos_uvm_gf
python test_uvmnet.py --input ../datasets/foggy_photos_upright --checkpoint checkpoints_ponmudi_gf/uvmnet_combined_final.pt --output ../datasets/foggy_photos_uvm_gf --tile_size 1536 > ../datasets/foggy_photos_uvm_gf/dehaze_log.txt 2>&1
echo "full-size dehazed: $(ls ../datasets/foggy_photos_uvm_gf/*_restored.png | wc -l) photos at $(date +%T)"
python foggy_ft/shootout/gf_scale_test2.py 2>&1 | grep -v -E "^done|Warning|warn"
echo GF_RUN_FINISHED
