#!/bin/bash
# Every model of the project on C:\projects\jewwww\finalin\foggyimages, then the gallery in finalout\gallery_foggyimages.  (resumes where it stopped)
S=/mnt/c/projects/jewwww/uvm/foggy_ft/newfolder
echo "[$(date +%T)] 1/5 setup (upright copies, 1536 px inputs)";            bash $S/gal_setup.sh 2>&1 | tail -2
echo "[$(date +%T)] 2/5 the 1536 px family (CLAHE, FogRemoval, DehazeSNN x4, UVM models, averaging tests)"; bash $S/gal_run_1536.sh 2>&1 | grep -v -i warn | tail -22
echo "[$(date +%T)] 3/5 HazeFlow";                                             bash $S/gal_hazeflow.sh 2>&1 | grep -v -i warn | tail -3
echo "[$(date +%T)] 4/5 the full-size UVM family and all newer models";        bash $S/gal_fullsize.sh 2>&1 | grep -v -i warn
echo "[$(date +%T)] 5/5 gallery";                                              bash $S/gal_build.sh 2>&1 | grep -v -i warn
echo "[$(date +%T)] ALL_DONE"
