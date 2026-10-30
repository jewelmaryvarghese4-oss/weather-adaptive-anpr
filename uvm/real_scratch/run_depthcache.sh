#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
python real_scratch/depthfog_cache.py --limit ${1:-0} 2>&1 | tr '\r' '\n' | grep -a -v -i warn | tail -${2:-6}
