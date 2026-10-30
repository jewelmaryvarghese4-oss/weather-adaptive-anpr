#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
python real_scratch/depthfog_calibrate.py 2>&1 | grep -a -v -i warn
