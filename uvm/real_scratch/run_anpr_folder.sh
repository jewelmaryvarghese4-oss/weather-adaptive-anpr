#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
python real_scratch/run_anpr_folder.py --input "$1" --out "$2" 2>&1 | tr '\r' '\n' | grep -a -v -i warn | tail -14
