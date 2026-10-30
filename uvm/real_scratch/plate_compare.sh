#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm/foggy_ft/shootout
python prep_new_sources.py
export PYTHONPATH=~/dehaze-eval/extra
cd ~/dehaze-eval
python /mnt/c/projects/jewwww/uvm/foggy_ft/shootout/shoot_score_select2_cpu.py raw uvm_old+gf uvm_ft2+gf uvm_ponmudi+gf uvm_gf+gf uvm_gp+gf uvm_gpt new_ep22 new_ep22+gf hybrid blend5050 inv100 phys_normal phys_strong_exp phys_gopro_se 2>&1 | grep -v Warning
echo PLATE_COMPARE_DONE
