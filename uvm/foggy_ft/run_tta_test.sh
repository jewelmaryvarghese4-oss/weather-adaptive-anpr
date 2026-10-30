#!/bin/bash
# Waits for the filter-trained run to finish, then tests flip / multi-scale averaging at 1536 px. New outputs only (~/dehaze-eval/out/uvm_gf_*).
cd /mnt/c/projects/jewwww/uvm
idle=0
until grep -aq "GF_RUN_FINISHED" logs/gf_chain.log 2>/dev/null; do
  if ps -eo args | grep -E "train_uvm_gf|shoot_|test_uvmnet|gf_scale|run_gf" | grep -vq grep; then idle=0; else idle=$((idle+1)); fi
  if [ $idle -ge 5 ]; then echo "TTA_WAIT_ABORTED: the previous run is not running and never finished"; exit 1; fi
  sleep 60
done
echo "previous run finished at $(date +%T); starting the averaging test"
source ~/mamba-venv/bin/activate
cd ~/dehaze-eval
export PYTHONPATH=~/dehaze-eval/extra
for m in flip flip4 ms msflip; do
  python /mnt/c/projects/jewwww/uvm/foggy_ft/shootout/shoot_dehaze_tta.py --model uvm_gf --mode $m > logs/tta_$m.log 2>&1
  tail -1 logs/tta_$m.log
done
python /mnt/c/projects/jewwww/uvm/foggy_ft/shootout/shoot_score_select2.py raw uvm_gf+gf uvm_gf_flip+gf uvm_gf_flip4+gf uvm_gf_ms+gf uvm_gf_msflip+gf 2>&1 | grep -v Warning
echo TTA_TEST_FINISHED
