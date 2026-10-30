#!/bin/bash
# Starts the FOGGY plate reader web page (own port 8001, own runs folder app_runs_foggy/). From Windows:  Start_Foggy_Plate_Reader.bat
# then open http://localhost:8001
for pid in $(pgrep -f "python foggy_plate_app.py"); do kill "$pid" 2>/dev/null; done   # an old copy would keep the port busy
sleep 1
source ~/mamba-venv/bin/activate
unset PYTORCH_CUDA_ALLOC_CONF
cd /mnt/c/projects/jewwww/uvm
exec python foggy_plate_app.py "$@"
