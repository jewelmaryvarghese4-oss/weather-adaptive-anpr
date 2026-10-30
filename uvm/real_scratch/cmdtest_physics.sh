#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
W=../datasets/finalrun_foggyimages
python foggy_ft/clean/physics_dehaze.py --source $W/raw --hazy $W/upright --output $W/_cmdtest --smooth 0.03 --gamma 1.25 --only 1 | tail -1
python foggy_ft/clean/exposure_match.py --input $W/_cmdtest --output $W/_cmdtest_exp
ls $W/_cmdtest_exp
