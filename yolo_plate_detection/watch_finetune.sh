#!/bin/bash
# Live view of finetune_traffic.py training. Refreshes every 20 s; Ctrl+C stops watching only
# (the training itself keeps running).
#   bash watch_finetune.sh [run name, default traffic_1280]
DIR=/mnt/c/projects/jewwww/yolo_plate_detection
RUN=${1:-traffic_1280}
CSV=$DIR/runs_ft/$RUN/results.csv
LOG=$DIR/finetune_traffic.log
while true; do
  clear
  echo "=== detector fine-tune '$RUN'   $(date '+%H:%M:%S')   (Ctrl+C = stop watching, training continues)"
  if pgrep -f "finetune_traffic.py train" > /dev/null; then echo "status: RUNNING"; else echo "status: not running (finished or stopped)"; fi
  echo
  echo "--- current epoch ---"
  echo "  epoch    GPU_mem  box_loss  cls_loss  dfl_loss  plates  size   progress"
  tr '\r' '\n' < "$LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E '^ +[0-9]+/[0-9]+ ' | tail -n 1
  tr '\r' '\n' < "$LOG" 2>/dev/null | grep -E 'Error|error|out of memory' | tail -n 3
  echo
  echo "--- finished epochs (validation on the 133-image small-plate set) ---"
  if [ -f "$CSV" ]; then
    awk -F, 'NR==1{printf "%6s %10s %8s %7s %9s %9s\n","epoch","precision","recall","mAP50","mAP50-95","box_loss"}
             NR>1{printf "%6d %10.3f %8.3f %7.3f %9.3f %9.3f\n",$1,$6,$7,$8,$9,$3}' "$CSV" | tail -n 16
  else
    echo "(no finished epoch yet)"
  fi
  echo
  echo "--- GPU ---"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
  sleep 20
done
