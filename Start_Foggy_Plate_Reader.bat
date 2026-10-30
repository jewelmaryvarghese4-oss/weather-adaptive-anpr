@echo off
title Foggy Plate Reader (close this window to stop the server)
echo Starting the foggy plate reader. The browser opens in about 35 seconds.
echo Leave this window open while you use it; close it to stop.
start "" cmd /c "timeout /t 35 /nobreak >nul & start http://localhost:8001"
wsl.exe bash /mnt/c/projects/jewwww/uvm/start_foggy_plate_app.sh
pause
