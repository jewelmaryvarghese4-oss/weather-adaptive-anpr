#!/bin/bash
# After the depth-fog training (SEPARATE model): candidates on the 60 real Ponmudi photos, then option 4 (video plate tracks + multi-frame fusion).
# Everything goes to NEW folders; nothing existing is changed.
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7
for pair in "18:best_val_ema.pt" "14:keep_epoch14_ema.pt" "20:keep_epoch20_ema.pt"; do
  n=${pair%%:*}; ck=${pair#*:}
  echo "== depth-fog epoch $n =="
  python real_scratch/dehaze_photos.py --checkpoint checkpoints_depthfog/$ck \
    --input ../datasets/foggy_photos_upright --output ../datasets/foggy_photos_uvm_depthfog_ep$n 2>&1 | tr '\r' '\n' | tail -1
done
echo DEHAZE_DEPTHFOG_DONE
echo "== option 4: plate tracks from the videos =="
python foggy_ft/fusion/fusion_tracks.py 2>&1 | tr '\r' '\n' | grep -a -E "^video|FUSION_TRACKS_DONE|Error|Traceback"
echo "== option 4: fusion and reading =="
python foggy_ft/fusion/fusion_fuse.py 2>&1 | grep -a -v -i warn
echo POST_GPU_DONE
