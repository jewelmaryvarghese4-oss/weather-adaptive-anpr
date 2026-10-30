#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm/foggy_ft/shootout
python prep_new_sources2.py
export PYTHONPATH=~/dehaze-eval/extra
cd ~/dehaze-eval
python /mnt/c/projects/jewwww/uvm/foggy_ft/shootout/shoot_score_select2_cpu.py raw uvm_gp+gf new_ep22 phys_strong_exp new_long blend_long phys_long_normal phys_long_se 2>&1 | grep -v Warning
python /mnt/c/projects/jewwww/uvm/foggy_ft/shootout/fog_metrics_all.py 2>/dev/null | grep -E "folder|^raw |uvm_gp |new_ep22|phys_strong_exp|new_long|blend_long|phys_long"
echo PLATE_COMPARE2_DONE
