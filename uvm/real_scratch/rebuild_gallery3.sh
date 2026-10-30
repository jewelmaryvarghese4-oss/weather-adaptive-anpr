#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
python foggy_ft/gallery/build_gallery.py --only f_inv 2>&1 | tr '\r' '\n' | tail -2
echo GALLERY3_DONE
