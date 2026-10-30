#!/bin/bash
# Starts the plate reader web page. From Windows:  Start_Plate_Reader.bat   (or  wsl.exe bash /mnt/c/projects/jewwww/uvm/start_plate_app.sh)
# then open http://localhost:8000
for pid in $(pgrep -f "python plate_app.py"); do kill "$pid" 2>/dev/null; done   # an old copy from an earlier launch would keep the port busy
sleep 1
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
exec python plate_app.py "$@"
