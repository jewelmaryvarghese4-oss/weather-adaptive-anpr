#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
for v in f_hrb f_roh; do python foggy_ft/gallery/build_gallery.py --only $v 2>&1 | tr '\r' '\n' | tail -2; done
echo GALLERY2_DONE
