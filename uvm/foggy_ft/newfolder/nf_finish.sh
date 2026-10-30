#!/bin/bash
# waits for the full-size family, then plate reading, the gallery and the plates page
while ! grep -aq NF_FULL_FINISHED /mnt/c/projects/jewwww/uvm/logs/nf_rest.log 2>/dev/null; do sleep 10; done
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
echo "plates start $(date +%T)"
HOME=/home/jewel/nfhome python foggy_ft/newfolder/nf_plates.py 2>&1 | grep -v -E "Warning|WARN"
python foggy_ft/newfolder/nf_plates_page.py
export NF_RESULTS=/mnt/c/projects/jewwww/datasets/newfoggyphotos/results NF_TITLE="New foggy photos"
python foggy_ft/gallery/build_gallery.py --procs 2 2>&1 | tail -3
cp /mnt/c/projects/jewwww/datasets/gallery/index.html $NF_RESULTS/gallery/index.html
cp /mnt/c/projects/jewwww/datasets/gallery/Open_Gallery.bat $NF_RESULTS/gallery/Open_Gallery.bat
echo NF_FINISHED
