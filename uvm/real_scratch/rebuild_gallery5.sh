#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
for v in f_pg f_pgs f_pgse; do python foggy_ft/gallery/build_gallery.py --only $v 2>&1 | tr '\r' '\n' | tail -1; done
echo GALLERY5_DONE
