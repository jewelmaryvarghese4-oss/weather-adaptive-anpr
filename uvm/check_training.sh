#!/bin/bash
# Shows the progress of the compare_rj250 training run (all seeds) and the final summary when it is done.
cd /mnt/c/projects/jewwww/uvm
echo "training processes running: $(ps -eo args | grep -E 'train_lprnet|compare_two_line' | grep -v grep | wc -l)"
for d in checkpoints_compare_rj250/*; do
  echo "== $d"
  tr '\r' '\n' < "$d/train.log" | grep -E '^epoch +[0-9]+ \|' | tail -n 2
done
echo "--- run log (seed results and final summary appear here)"
grep -E '^\[train\]|held-out exact|summary|^fixed|layout' compare_rj250.log | tail -n 12
