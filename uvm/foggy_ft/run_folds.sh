#!/bin/bash
# Trains one reader per fold (weather augmentation + the foggy crops of the OTHER plates) into checkpoints_foggy_ft/ (new folder;
# the installed reader and all earlier checkpoints are not touched), then scores every fold on its own held-out plates.
#   bash foggy_ft/run_folds.sh [tag]        (about 25 min per fold; run from /mnt/c/projects/jewwww/uvm)
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
TAG=${1:-verified}
export TWO_LINE_MODE=fixed
for k in 0 1 2 3; do
  python compare_two_line_modes.py --modes fixed --seeds 0 --epochs 40 --avg_last 10 --out_root checkpoints_foggy_ft/${TAG}_fold$k \
    --captcha_root /mnt/c/projects/jewwww/datasets/lprnet_data --test_dir ../datasets/heldout_rj250 \
    --extra "--data_root ../datasets/syntheticcrops --manifest ../datasets/foggy_ft_manifest_${TAG}_fold$k.csv --heavy_repeats --synthetic_degrade_prob 0.5 --synthetic_mix 2000 --degrade_prob 0.5 --weather_prob 0.5" \
    2>&1 | tee logs/foggy_ft_${TAG}_fold$k.log
done
python foggy_ft/eval_folds.py $TAG
