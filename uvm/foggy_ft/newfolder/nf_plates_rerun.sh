#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
rm -rf /mnt/c/projects/jewwww/datasets/newfoggyphotos/results/plates/crops
HOME=/home/jewel/nfhome python foggy_ft/newfolder/nf_plates.py 2>&1 | grep -v -E "Warning|WARN"
python foggy_ft/newfolder/nf_plates_page.py
echo PLATES_RERUN_FINISHED
