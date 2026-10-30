"""Tidies C:/projects/jewwww. Nothing is deleted: folders and files are only moved, and every move is written to
organize_log.json so  python organize_project.py --undo  puts everything back.

    python organize_project.py            # dry run: prints what would move
    python organize_project.py --apply    # does it
    python organize_project.py --undo     # reverses the last --apply

Folders that code reads by a fixed relative path (datasets, evaluation, UVM_TRAFFIC_DATASET, uvm/checkpoints_lprnet,
uvm/checkpoints_lprnet_synth, uvm/checkpoints_combined, uvm/checkpoints_traffic, yolo_plate_detection, venv) stay where they are.
"""
import glob
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "organize_log.json")


def P(*a):
    return os.path.join(ROOT, *a)


moves = []   # (source, destination folder)

# ---- root: documents and old dehazing work ----------------------------------------------------------
for f in ("ANPR_Results_Chapter.docx", "Dehazing_Results_Report.docx", "Project_Diary_Action_Plan_updated.docx",
          "Weather Adaptive - updated.pptx", "Weather Adaptive.pptx", "ANPR_Intermediate_Review.pptx", "ANPR_Intermediate_Review_teal.pptx"):
    moves.append((P(f), P("documents")))
moves.append((P("Fundamentals of Digital Image Processing.pdf"), P("documents", "reference")))
for d in ("dcp_dehazing", "clahe_dehazing", "neural_dehazing", "mamba"):
    moves.append((P(d), P("dehazing_methods")))
moves.append((P("misc_legacy"), P("archive")))
moves.append((P("dehazed_out"), P("archive")))
moves.append((P("plate_dehaze_fix_review"), P("archive")))
moves.append((P("mambazip.zip"), P("archive")))
moves.append((P("checkpoints"), P("archive")))          # the five early Mamba dehazer checkpoints

# ---- uvm: logs, backups, old outputs, old experiment checkpoints ------------------------------------
U = lambda *a: P("uvm", *a)
for f in glob.glob(U("*.log")):
    if os.path.basename(f) != "compare_rj250.log":      # check_training.sh reads this one
        moves.append((f, U("logs")))
for f in glob.glob(U("*.bak_*")):
    moves.append((f, U("backups")))
for f in ("live_plates_log.csv", "pipeline_ocr_results.csv"):
    moves.append((U(f), U("old_outputs")))
for d in sorted(glob.glob(U("test_outputs_*"))):
    moves.append((d, U("test_outputs_archive")))
for d in ("ccpd_preview", "domain_check", "native_test", "plate_crops", "synthetic_plate_preview", "val_check_100"):
    moves.append((U(d), U("previews_and_checks")))
for d in ("compare", "compare_degrade", "compare_ds11", "compare_heavy", "compare_heavy2", "compare_heavy3", "compare_twoline",
          "compare_videoocr", "compare_zen", "compare_zen_verified", "final", "final_zen", "final_zen_v2", "lowlight",
          "lprnet_hybrid", "lprnet_mixed", "lprnet_trim", "plates", "plates_v3_densityaug"):
    moves.append((U("checkpoints_" + d), U("checkpoints_experiments")))


def apply():
    done = []
    for src, dst_dir in moves:
        if not os.path.exists(src):
            continue
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.exists(dst):
            print("SKIP (already exists):", os.path.relpath(dst, ROOT))
            continue
        os.makedirs(dst_dir, exist_ok=True)
        try:
            shutil.move(src, dst)
        except OSError as e:
            print("SKIP (in use or blocked):", os.path.relpath(src, ROOT), "-", e)
            continue
        done.append([src, dst])
        print("moved", os.path.relpath(src, ROOT), "->", os.path.relpath(dst_dir, ROOT) + os.sep)
    json.dump(done, open(LOG, "w"), indent=1)
    print(f"\n{len(done)} items moved; undo with: python organize_project.py --undo")


def undo():
    for src, dst in reversed(json.load(open(LOG))):
        if os.path.exists(dst) and not os.path.exists(src):
            shutil.move(dst, src)
            print("restored", os.path.relpath(src, ROOT))


if __name__ == "__main__":
    if "--undo" in sys.argv:
        undo()
    elif "--apply" in sys.argv:
        apply()
    else:
        n = 0
        for src, dst_dir in moves:
            if os.path.exists(src):
                n += 1
                print("would move", os.path.relpath(src, ROOT), "->", os.path.relpath(dst_dir, ROOT) + os.sep)
        print(f"\n{n} items would move (dry run; add --apply to do it)")
