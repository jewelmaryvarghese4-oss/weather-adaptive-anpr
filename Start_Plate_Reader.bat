@echo off
title Plate Reader (close this window to stop the server)
echo Starting the plate reader. The browser opens in about 25 seconds.
echo Leave this window open while you use it; close it to stop.
start "" cmd /c "timeout /t 25 /nobreak >/dev/null & start http://localhost:8000"
wsl.exe bash /mnt/c/projects/jewwww/uvm/start_plate_app.sh
pause
