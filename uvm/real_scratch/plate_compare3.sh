#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm/foggy_ft/shootout
python prep_new_sources3.py
export PYTHONPATH=~/dehaze-eval/extra
cd ~/dehaze-eval
python /mnt/c/projects/jewwww/uvm/foggy_ft/shootout/shoot_score_select2_cpu.py raw uvm_gp+gf phys_strong_exp new_long df_ep14 df_ep18 df_ep20 df_blend phys_df_blend_se phys_df_se 2>&1 | grep -v Warning
python /mnt/c/projects/jewwww/uvm/foggy_ft/shootout/fog_metrics_all.py 2>/dev/null | grep -E "^folder|^raw |uvm_gp |phys_strong_exp|new_long|df_|hybrid "
echo PLATE_COMPARE3_DONE
