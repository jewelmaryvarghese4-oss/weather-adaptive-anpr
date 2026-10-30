# Fresh real-only UVM: commands (WSL, one block at a time, each shows a live progress bar)

## 0. Keep the PC awake (Windows power settings: sleep = never), open WSL
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm

## 1. Extract the frames (about 10-20 min, about 13 GB). Live bar: frames, speed, time left, videos done
    python real_scratch/extract_frames.py --step 2 --workers 4 2>&1 | tee ../datasets/real_scratch_extract.log

## 2. Train (about 5 h for 30 epochs). Live bar per epoch: step, speed, time left, loss, learning rate
    PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7 python real_scratch/train_uvm_real.py --root ../datasets/real_scratch --out_dir checkpoints_real_scratch --epochs 30 --samples 10000 2>&1 | tee -a ../datasets/real_scratch_train.log
    # keep that terminal open. If it stops (Ctrl+C, closed window, sleep) run the SAME command again: it resumes from the last finished epoch.
    # after every epoch it prints no-reference numbers on held-out GoPro test videos and saves checkpoints_real_scratch/preview_epochN.jpg (foggy | output)

## 3. Dehaze your photos (live bar per photo)
    python real_scratch/dehaze_photos.py --checkpoint checkpoints_real_scratch/real_final_ema.pt --input ../datasets/foggy_photos_upright --output ../datasets/foggy_photos_uvm_real
    python real_scratch/dehaze_photos.py --checkpoint checkpoints_real_scratch/real_final_ema.pt --input ../datasets/newfoggyphotos/results/upright --output ../datasets/newfoggyphotos/results/full/real

## 4. Add it to the galleries
    python foggy_ft/gallery/build_gallery.py --procs 2
    NF_RESULTS=/mnt/c/projects/jewwww/datasets/newfoggyphotos/results NF_TITLE="New foggy photos" python foggy_ft/gallery/build_gallery.py --procs 2
