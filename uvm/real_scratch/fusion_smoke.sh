#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
python foggy_ft/fusion/fusion_tracks.py --out ../datasets/fusion_smoke --limit_videos 1 --max_frames 120 --device cpu 2>&1 | tr '\r' '\n' | grep -a -v -i warn | tail -4
python foggy_ft/fusion/fusion_fuse.py --fusion ../datasets/fusion_smoke --device cpu 2>&1 | grep -a -v -i warn | tail -14
echo FUSION_SMOKE_DONE
