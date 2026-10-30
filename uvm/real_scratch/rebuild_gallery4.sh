#!/bin/bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
for v in f_pn f_ps f_pse; do python foggy_ft/gallery/build_gallery.py --only $v 2>&1 | tr '\r' '\n' | tail -1; done
echo GALLERY4_DONE
