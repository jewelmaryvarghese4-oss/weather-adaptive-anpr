#!/bin/bash
# GoProHazy matched frames (lossless), trained WITHOUT a pixel loss against the clear frame (they are not pixel-aligned), on top of the filter-trained UVM.
# New outputs only: datasets/gopro_hazy/pairs, checkpoints_ponmudi_gopro/, logs/gp_*, ~/dehaze-eval/out/uvm_gp.
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python foggy_ft/gopro/patch_shoot_gp.py
G=../datasets/gopro_hazy
[ -s $G/pairs/pairs.csv ] || python foggy_ft/gopro/extract_pairs.py --per_video 50 --hold 7 --hold_per_video 14
R=../datasets/1001hazeimages
echo "training starts $(date +%T)"
python train_uvm_gopro.py --ots_root $R/OTS_BETA --plates_root $R/FoggyPlates --nhhaze_root $R/NH-HAZE \
  --synth_clear_root ../datasets/rainfog_clear $R/FoggyPlates/clear --synth_samples 1000 --ponmudi_samples 2000 \
  --ots_limit 1000 --nhhaze_repeat 10 --lowlight_prob 0.15 --lr 1e-5 --epochs 4 --ckpt_every 4 --gf_radius 4 --gf_eps 0.01 \
  --gopro_root $G --gopro_lambda 0.1 --gopro_batch 3 --gopro_crop 288 \
  --init_ckpt checkpoints_ponmudi_gf/uvmnet_combined_final.pt --ckpt_dir checkpoints_ponmudi_gopro > logs/gp_uvm_ft.log 2>&1
tr '\r' '\n' < logs/gp_uvm_ft.log | grep -a -E "GoProHazy|GoPro unaligned|^epoch +[0-9]+ \|" | cut -c1-220
[ -f checkpoints_ponmudi_gopro/uvmnet_combined_final.pt ] || { echo "GP_TRAINING_FAILED"; tr '\r' '\n' < logs/gp_uvm_ft.log | grep -a -v Warning | tail -5 | cut -c1-200; exit 1; }
echo "training finished $(date +%T); dehazing the held-out GoPro frames"
rm -rf $G/heldout_hazy $G/out_before $G/out_after; mkdir -p $G/heldout_hazy $G/out_before $G/out_after
cp $G/pairs/heldout/*_hazy.png $G/heldout_hazy/
python test_uvmnet.py --input $G/heldout_hazy --checkpoint checkpoints_ponmudi_gf/uvmnet_combined_final.pt --output $G/out_before --tile_size 1536 > logs/gp_before.log 2>&1
python test_uvmnet.py --input $G/heldout_hazy --checkpoint checkpoints_ponmudi_gopro/uvmnet_combined_final.pt --output $G/out_after --tile_size 1536 > logs/gp_after.log 2>&1
python foggy_ft/gopro/eval_heldout.py 2>&1 | grep -v Warning
echo "dehazing the 50 Ponmudi photos at 1536 px, $(date +%T)"
cd ~/dehaze-eval
export PYTHONPATH=~/dehaze-eval/extra
~/mamba-venv/bin/python scripts/shoot_dehaze.py --model uvm_gp > logs/uvm_gp.log 2>&1
tail -1 logs/uvm_gp.log
~/mamba-venv/bin/python /mnt/c/projects/jewwww/uvm/foggy_ft/shootout/shoot_score_select2.py raw uvm_gf+gf uvm_gp+gf 2>&1 | grep -v Warning
echo GP_RUN_FINISHED
