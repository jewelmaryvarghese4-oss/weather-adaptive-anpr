#!/bin/bash
# Builds the gallery of the new photo folder and puts it into finalout/gallery_foggyimages
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
R=/mnt/c/projects/jewwww/datasets/finalrun_foggyimages_gallery/results
OUT=/mnt/c/projects/jewwww/finalout/gallery_foggyimages
export NF_RESULTS=$R NF_TITLE="Foggy images (finalin/foggyimages): every model"
python foggy_ft/gallery/build_gallery.py --procs 3 2>&1 | tail -4
cp /mnt/c/projects/jewwww/datasets/gallery/index.html $R/gallery/index.html
cp /mnt/c/projects/jewwww/datasets/gallery/Open_Gallery.bat $R/gallery/Open_Gallery.bat
if [ -e "$OUT" ]; then echo "target exists, not overwriting: $OUT"; else mkdir -p /mnt/c/projects/jewwww/finalout && mv $R/gallery $OUT && echo "gallery moved to $OUT"; fi
echo GALLERY_BUILD_FINISHED
