#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
export NF_RESULTS=/mnt/c/projects/jewwww/datasets/newfoggyphotos/results NF_TITLE="New foggy photos"
python foggy_ft/gallery/build_gallery.py --procs 2 2>&1 | tail -2
cp /mnt/c/projects/jewwww/datasets/gallery/index.html $NF_RESULTS/gallery/index.html
echo REBUILT
