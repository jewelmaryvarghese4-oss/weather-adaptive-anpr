#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
for v in f_df18 f_dfps; do python foggy_ft/gallery/build_gallery.py --only $v 2>&1 | tr '\r' '\n' | tail -1; done
echo GALLERY6_DONE
