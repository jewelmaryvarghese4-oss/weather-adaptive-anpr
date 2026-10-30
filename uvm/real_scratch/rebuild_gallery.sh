#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
python foggy_ft/gallery/build_gallery.py --only f_real 2>&1 | tr '\r' '\n' | tail -3
echo GALLERY_DONE
