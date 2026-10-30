#!/bin/bash
source ~/mamba-venv/bin/activate
unset PYTORCH_CUDA_ALLOC_CONF
cd /mnt/c/projects/jewwww/uvm
python real_scratch/run_video_pipeline.py --input "$1" --out "$2" ${3:+--max_frames $3} 2>&1 | tr '\r' '\n' | grep -a -v -i warn | tail -${4:-12}
