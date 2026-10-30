#!/bin/bash
# 4-fold training with the session crops added; writes ONLY to checkpoints_session_ft/ and logs/ (new). Run from anywhere in WSL.
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
export TWO_LINE_MODE=fixed
for k in 0 1 2 3; do
  python compare_two_line_modes.py --modes fixed --seeds 0 --epochs 40 --avg_last 10 --out_root checkpoints_session_ft/session_fold$k \
    --captcha_root /mnt/c/projects/jewwww/datasets/lprnet_data --test_dir ../datasets/heldout_rj250 \
    --extra "--data_root ../datasets/syntheticcrops --manifest ../datasets/foggy_ft_manifest_session_fold$k.csv --heavy_repeats --synthetic_degrade_prob 0.5 --synthetic_mix 2000 --degrade_prob 0.5 --weather_prob 0.5" \
    2>&1 | tee logs/session_ft_fold$k.log
done
python foggy_ft/eval_folds_session.py
