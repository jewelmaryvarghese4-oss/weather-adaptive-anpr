"""
Builds a self-contained zip of everything related to the LPRNet plate reader, to copy to another device.
    python foggy_ft/export_lprnet.py     ->  ../exports/lprnet_export_<date>.zip
Nothing in the project is changed; the zip is made from copies.
"""
import hashlib
import os
import shutil
import zipfile
import sys
from datetime import date

LITE = "--lite" in sys.argv        # smaller zip: skips earlier versions, no-weather seeds and fold models

UVM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.dirname(UVM)
D = os.path.join(ROOT, "datasets")
STAGE = os.path.join(ROOT, "exports", "_stage", "lprnet_export")
ZIP = os.path.join(ROOT, "exports", "lprnet_export%s_%s.zip" % ("_lite" if LITE else "", date.today().isoformat()))
shutil.rmtree(os.path.dirname(STAGE), ignore_errors=True)
for sub in ("src", "train", "weights", "detector", "data"):
    os.makedirs(os.path.join(STAGE, sub))


def cp(src, dst_rel):
    dst = os.path.join(STAGE, dst_rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


# --- code to READ plates (everything predict_plate.py and evaluate.py need)
for f in ("model_lprnet.py", "lprnet_vocab.py", "plate_format.py", "two_line_plate.py"):
    cp(os.path.join(UVM, f), "src/" + f)
# --- code to TRAIN / evaluate / augment
for f in ("train_lprnet.py", "dataset_lprnet.py", "dataset_manifest.py", "synthetic_plate_dataset.py", "degrade.py", "weather_crop.py",
          "synth_rainfog.py", "compare_two_line_modes.py", "label_review.py", "clahe_preprocess.py", "dataset_captcha_style.py",
          "eval_video_benchmark.py"):
    if os.path.exists(os.path.join(UVM, f)):
        cp(os.path.join(UVM, f), "train/" + f)
cp(os.path.join(ROOT, "evaluation", "baseline_ocr", "ground_truth.json"), "train/ground_truth.json")
cp(os.path.join(UVM, "foggy_ft", "make_folds.py"), "train/make_folds.py")

# --- weights (LPRNet is ~2 MB each)
W = [("checkpoints_lprnet/lprnet_avg.pt", "weights/lprnet_avg_INSTALLED.pt"),
     ("checkpoints_lprnet_synth/lprnet_synth_final.pt", "weights/lprnet_synth_pretrain_final.pt")]
for f in sorted(os.listdir(os.path.join(UVM, "checkpoints_lprnet"))):
    if not LITE and f.startswith("lprnet_avg_") and f.endswith(".pt"):
        W.append(("checkpoints_lprnet/" + f, "weights/earlier_versions/" + f))
for s in (0, 1, 2):
    W.append(("checkpoints_compare_weather/fixed_s%d/lprnet_avg.pt" % s, "weights/weather_trained/lprnet_weather_seed%d.pt" % s))
    if not LITE:
        W.append(("checkpoints_compare_rj250/fixed_s%d/lprnet_avg.pt" % s, "weights/same_recipe_no_weather/lprnet_noweather_seed%d.pt" % s))
for k in ([] if LITE else range(4)):
    W.append(("checkpoints_foggy_ft/verified_fold%d/fixed_s0/lprnet_avg.pt" % k, "weights/foggy_crops_fold_models/lprnet_foggyfold%d.pt" % k))
for src, dst in W:
    p = os.path.join(UVM, src)
    if os.path.exists(p):
        cp(p, dst)
    else:
        print("missing (skipped):", src)

# --- plate detector (optional, for whole photos)
cp(os.path.join(ROOT, "backups", "working_state_2026-10-18", "models", "yolo_plate_detector_15k_best.pt"), "detector/yolo_plate_detector_15k_best.pt")

# --- small test sets (images + labels) and the verified foggy labels
cp(os.path.join(D, "heldout_rj250"), "data/heldout_rj250")
cp(os.path.join(D, "foggy_crops_test"), "data/foggy_crops_test")
os.makedirs(os.path.join(STAGE, "data", "kerala16"))
cp(os.path.join(D, "ocrtrain", "labels.json"), "data/kerala16/labels.json")
cp(os.path.join(D, "ocrtrain", "crops"), "data/kerala16/crops")
cp(os.path.join(D, "foggy_ft", "foggy_labels_verified.csv"), "data/foggy_labels_verified.csv")
cp(os.path.join(D, "foggy_ft", "crops"), "data/foggy_verified_crops")

# --- usage files
here = os.path.dirname(os.path.abspath(__file__))
for f in ("EXPORT_README.md", "export_predict_plate.py", "export_evaluate.py", "export_requirements.txt"):
    dst = {"EXPORT_README.md": "README.md", "export_predict_plate.py": "predict_plate.py", "export_evaluate.py": "evaluate.py",
           "export_requirements.txt": "requirements.txt"}[f]
    cp(os.path.join(here, f), dst)

# --- checksums and the zip
lines = []
for base, _, files in os.walk(STAGE):
    for f in sorted(files):
        p = os.path.join(base, f)
        lines.append("%s  %s" % (hashlib.md5(open(p, "rb").read()).hexdigest(), os.path.relpath(p, STAGE).replace(os.sep, "/")))
open(os.path.join(STAGE, "MANIFEST.md5"), "w").write("\n".join(sorted(lines, key=lambda s: s.split("  ")[1])) + "\n")
if os.path.exists(ZIP):
    os.remove(ZIP)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for base, _, files in os.walk(STAGE):
        for f in files:
            p = os.path.join(base, f)
            z.write(p, os.path.join("lprnet_export", os.path.relpath(p, STAGE)))
print("files: %d | zip: %s | %.1f MB" % (len(lines), ZIP, os.path.getsize(ZIP) / 1e6))
