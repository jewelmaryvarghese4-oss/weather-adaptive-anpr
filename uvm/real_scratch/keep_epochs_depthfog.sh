#!/bin/bash
# Keep every epoch's EMA weights of the depth-fog run (the run itself only saves every 4th epoch + the best-by-score one). Read-only on the training files.
d=/c/projects/jewwww/uvm/checkpoints_depthfog
for n in 10 11 12 13 14 15 16 17 18 19 20; do
  until [ -f $d/preview_epoch$n.jpg ]; do
    wsl.exe -d Ubuntu -- pgrep -f train_uvm_depthfog.py >/dev/null 2>&1 || { echo "training ended"; exit 0; }
    sleep 20
  done
  sleep 40
  cp $d/real_latest_ema.pt $d/keep_epoch${n}_ema.pt && echo "kept epoch $n"
done
