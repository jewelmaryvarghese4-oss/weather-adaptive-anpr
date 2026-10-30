> **Note:** this is the original folder map of the working machine. It lists folders (datasets, checkpoints, logs, archive) that are not part of this repository. See the top-level README for what is here.

# Weather Adaptive ANPR (jewwww)

YOLOv11 plate detector -> weather restoration (CLAHE / UVMNet) -> LPRNet plate reader, run in real time on traffic video.

## Folder map

| Folder | What is in it |
|---|---|
| `uvm/` | **The active code.** Plate reading (`read_plate.py`, `live_plates.py`, `read_video.py`), reader training (`train_lprnet.py`, `compare_two_line_modes.py`, `video_label_review.py`, `label_review.py`), dehazing (`test_uvmnet.py`, `dispatch_dehaze.py`, `train_combined.py`), the web interface (`plate_app.py`, `plate_app.html`) |
| `uvm/checkpoints_lprnet/` | **The reader in use**: `lprnet_avg.pt` (round 3, "rj250"). Older defaults are kept next to it as `lprnet_avg_pre_*.pt` (go back with `cp lprnet_avg_pre_rj250.pt lprnet_avg.pt`) |
| `uvm/checkpoints_combined/` | UVMNet weights in use (`uvmnet_combined_best_stateonly.pt`) |
| `uvm/checkpoints_lprnet_synth/`, `uvm/checkpoints_traffic/` | Starting points for training (synthetic-pretrained reader, traffic-pretrained UVMNet) |
| `uvm/checkpoints_compare_rj250/`, `..._videoocr2/` | The two most recent reader training runs (3 seeds each) |
| `uvm/checkpoints_experiments/` | All older training runs |
| `uvm/app_runs/` | One folder per run made in the browser interface: `plates.csv`, `tracks.csv`, plate crops, frame snapshots, unconfirmed tracks, annotated video |
| `uvm/logs/`, `uvm/backups/`, `uvm/old_outputs/` | Old training logs, code backups made before changes, old result files |
| `uvm/test_outputs/`, `uvm/test_outputs_archive/` | Dehazed test images (latest / older versions) |
| `uvm/previews_and_checks/` | Small preview and diagnostic folders |
| `yolo_plate_detection/` | Detector training, the 15k model (`runs/detect/runs/detect/anpr_plate_model_15k`), the fine-tuned model (`runs_ft`, not installed) |
| `evaluation/` | OCR ground truth (`baseline_ocr/ground_truth.json`, 218 scenes) and EasyOCR baseline results. Read by the training code, so it stays here |
| `datasets/` | All data. See `datasets/README.md` |
| `UVM_TRAFFIC_DATASET/` | Foggy Cityscapes traffic pairs used to train UVMNet |
| `dehazing_methods/` | First-stage dehazing work: `dcp_dehazing`, `clahe_dehazing`, `neural_dehazing` (transformer), `mamba` (first Mamba model) |
| `documents/` | Reports, diary, presentations, reference book |
| `archive/` | Old material kept just in case (`misc_legacy`, early checkpoints, old images, `mambazip.zip`) |
| `venv/` | Windows Python environment (the work runs in WSL's `~/mamba-venv`) |

## Everyday commands (run in WSL)

```bash
source ~/mamba-venv/bin/activate
cd /mnt/c/projects/jewwww/uvm
```

| Task | Command |
|---|---|
| Live plate reading on a video | `python live_plates.py --source <video> --save ../datasets/live_out/out.mp4 --log ../datasets/live_out/plates.csv` (add `--yolo_imgsz 1280` for 4K) |
| Read plates in photos | `python read_plate.py --image_dir <folder> --skip_dehaze --save_annotated <out>/annotated --save_crops_dir <out>/crops` |
| Dehaze a folder | `python test_uvmnet.py --input <folder> --checkpoint checkpoints_combined/uvmnet_combined_best_stateonly.pt --output <out>` |
| Browser interface | Double-click `Start_Plate_Reader.bat` in this folder (or `bash start_plate_app.sh` in WSL), then use http://localhost:8000. Pick videos or photos from your folders, watch plates appear live, and browse earlier runs. Results are saved in `uvm/app_runs/` |
| Several workers (no skipped frames) | add `--workers 4` to `live_plates.py`, or choose *Parallel workers* in the browser interface (default 4). Code: `uvm/parallel_reader.py`. Measured on this laptop: 1080p60 reaches about 62 frames/s with 4 workers (single process: 41), 4K about 20 (single: 13.5), because a 4K file can only be decoded at about 27 frames/s |
| Retrain the reader | see `uvm/logs/compare_videoocr2.log` and `compare_rj250.log` for the exact settings; check a run with `bash check_training.sh` |

## Tidying

`organize_project.py` (this folder) did the last tidy-up and logged every move in `organize_log.json`; `python organize_project.py --undo` puts everything back.
