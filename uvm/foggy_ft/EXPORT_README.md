# LPRNet plate reader — export (2026-10-18)

A small CNN (LPRNet, ~2 MB, input 200×24) that reads Indian licence plates from plate crops, with a YOLO plate detector for whole photos.
Everything here was copied from the project; nothing was changed. `weights/lprnet_avg_INSTALLED.pt` is the reader that was in use
(md5 044ae4514448…).

## Quick start (any computer with Python 3.10+)
```
pip install -r requirements.txt
python predict_plate.py --input data/foggy_crops_test/crops --no_detect                       # read plate crops
python predict_plate.py --input some_photo.jpg --detector detector/yolo_plate_detector_15k_best.pt --csv out.csv   # whole photo
python evaluate.py --test_dir data/foggy_crops_test                                           # score a checkpoint on labelled crops
python evaluate.py --test_dir data/heldout_rj250 --ckpt weights/weather_trained/lprnet_weather_seed1.pt
```
It runs on CPU if there is no GPU. `reading_format_corrected` is the one to use: it fixes letter/digit slips (O↔0, I↔1, …) using the
Indian plate layouts (LLDDLLDDDD, LLDDLDDDD, LLDLLDDDD, LLDDLLDDD, LLDLLLDDDD).

## What is in the folder
(The *lite* zip leaves out `weights/earlier_versions/`, `weights/same_recipe_no_weather/` and `weights/foggy_crops_fold_models/`; the full zip has them.)

| folder | content |
|---|---|
| `src/` | model and decoding code needed to read: `model_lprnet.py`, `lprnet_vocab.py`, `plate_format.py`, `two_line_plate.py` |
| `weights/` | `lprnet_avg_INSTALLED.pt` (in use) · `lprnet_synth_pretrain_final.pt` (start point for training) · `earlier_versions/` (previous defaults) · `weather_trained/` (3 seeds trained with fake rain+fog+grain, NOT installed) · `same_recipe_no_weather/` (3 seeds, same recipe without weather) · `foggy_crops_fold_models/` (4 fold models) |
| `detector/` | YOLO plate detector (15k-image model); use `--imgsz 1280` for large photos |
| `data/` | small test sets with `labels.json` + `crops/`: `heldout_rj250` (62), `foggy_crops_test` (49), `kerala16` (16); `foggy_labels_verified.csv` + `foggy_verified_crops/` (85 foggy plate crops with the plates you verified) |
| `train/` | training and augmentation code (`train_lprnet.py`, datasets, `weather_crop.py`, `synth_rainfog.py`, `compare_two_line_modes.py`, `make_folds.py`, `label_review.py`) and `ground_truth.json` |
| `MANIFEST.md5` | checksums of every file |

## Scores measured in this project (exact plates correct; "plain / format-corrected")
| checkpoint | heldout_rj250 (62) | foggy crops (49) | kerala16 (16)* | video benchmark (80 plates) |
|---|---|---|---|---|
| installed | 53 / 56 | 32 / 35 | 15 | 75 |
| weather seed 0 | 55 / 57 | 35 / 37 | 15 | 74 |
| weather seed 1 | 58 / 59 | 34 / 38 | 15 | 74 |
| weather seed 2 | 56 / 57 | 34 / 35 | 13 | 74 |

\* the installed model was trained on the Kerala 16, so that column only shows nothing broke.
On all 85 verified foggy crops, with each plate held out of training: installed 35 exact / 39 corrected, weather-trained 40 / 45.
The foggy labels are readings checked by one person on blurry photos, so treat small differences as noise. Weather-trained seeds are
better on fog and about equal on clear plates, a little lower on the clear video benchmark (frames right 77.6–78.2% vs 79.9%).

## Optional: smoothing view (`--view`, default `plain` = unchanged behaviour)
`predict_plate.py --view smooth|select` and `evaluate.py --view all`. The smooth view shrinks the crop to 64x16 and enlarges it to 128x32 (bicubic),
which removes grain. `select` reads the plain crop and the smoothed crop and keeps the reading that looks like a plate with the higher confidence
(no labels used). Idea and recipe from the separate "environment training / crop selection" experiments; tested here on this folder's test sets
(exact plates, format-corrected; plain / smooth / select):

| reader | foggy_crops_test (49) | heldout_rj250 (62, clear) | kerala16 (16, clear) |
|---|---|---|---|
| installed | 35 / 32 / 35 | 56 / 53 / 56 | 15 / 6 / 15 |
| weather seed 1 | 38 / 35 / **41** | 59 / 54 / 59 | - |

Use `select`, never `smooth` alone: the smoothed view alone hurts sharp plates a lot (kerala16: 15 -> 6). `select` kept every clear-plate score
unchanged and gave +3 on the foggy crops for the weather-trained reader, 0 for the installed one on these 49. On the larger set of 85 verified
foggy crops it was +2 (installed) and +3 to +4 (weather seeds); on the 22 of those crops that other experiments had held out of their training it was
+3 to +5 for every reader. Small samples: treat as a modest, consistent gain, not a guarantee. It makes reading about 2x slower per plate.

## Known weaknesses
Runs of three equal digits (…888 → 88), non-Kerala series and two-line plates on small bike/auto plates, plates under ~30 px wide, heavy
fog or glare. Dehazing the photo first (UVMNet, CLAHE, several others) did not improve reading on the foggy photos.

## Training (needs the training crops, which are NOT in this zip)
The reader was trained from `lprnet_synth_pretrain_final.pt` with `compare_two_line_modes.py --modes fixed --epochs 40 --avg_last 10` and
`--extra "--manifest <manifest.csv> --heavy_repeats --synthetic_degrade_prob 0.5 --synthetic_mix 2000 --degrade_prob 0.5 --weather_prob 0.5"`.
The manifest rows point to crop folders (Zenodo plates, video crops, hazy Rajasthan crops, synthetic crops) that live in the project's
`datasets/` folder; the scripts expect the project layout (`uvm/` next to `datasets/`). Copy those folders too if you want to retrain.
