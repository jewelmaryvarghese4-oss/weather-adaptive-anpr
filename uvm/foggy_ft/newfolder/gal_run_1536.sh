#!/bin/bash
# Every 1536 px dehazer on the new photos (results/s1536/<model>/<stem>.png). HOME is redirected so ~/dehaze-eval is the new-folder copy.
export HOME=/home/jewel/nfhome2
export PYTHONPATH=$HOME/dehaze-eval/extra
export TORCH_HOME=/home/jewel/.cache/torch HF_HOME=/home/jewel/.cache/huggingface
PY=/home/jewel/mamba-venv/bin/python
cd $HOME/dehaze-eval
S=/mnt/c/projects/jewwww/uvm/foggy_ft
for m in clahe fogremoval snn_l-reside-6k snn_l-reside-its snn_l-reside-ots snn_l-rs-haze uvm_old uvm_ft2 uvm_ponmudi uvm_gf uvm_pa uvm_gp; do
  $PY scripts/shoot_dehaze.py --model $m > logs/$m.log 2>&1
  echo "$m: $(ls out/$m/*.png 2>/dev/null | wc -l) photos | $(tail -1 logs/$m.log | cut -c1-100)"
done
for mode in flip flip4 ms msflip; do
  $PY $S/shootout/shoot_dehaze_tta.py --model uvm_gf --mode $mode > logs/tta_$mode.log 2>&1
  echo "tta $mode: $(tail -1 logs/tta_$mode.log | cut -c1-100)"
done
$PY $S/clean/tint_plates.py 2>&1 | grep -v Warning
echo NF_1536_FINISHED
