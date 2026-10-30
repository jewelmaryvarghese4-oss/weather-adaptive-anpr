cd ~/dehaze-eval
~/mamba-venv/bin/python - <<'EOF'
import glob, os
from PIL import Image
H = os.path.expanduser("~/dehaze-eval")
S = 1664 / 1536                       # measured: the output is the input magnified by 1.083 about the top-left corner
for tag, fill in (("hazeflow_a", "black"), ("hazeflow_b", "orig")):
    os.makedirs("%s/out/%s" % (H, tag), exist_ok=True)
for f in sorted(glob.glob(H + "/out/hazeflow/*.png")):
    stem = os.path.basename(f)
    o = Image.open(f).convert("RGB")
    ref = Image.open(H + "/data/in1536/" + stem).convert("RGB")
    w, h = o.size
    back = o.resize((round(w / S), round(h / S)), Image.LANCZOS)          # undo the magnification
    for tag, bg in (("hazeflow_a", Image.new("RGB", o.size, (0, 0, 0))), ("hazeflow_b", ref.copy())):
        bg.paste(back, (0, 0))
        bg.save("%s/out/%s/%s" % (H, tag, stem))
print("aligned outputs written; the bottom/right %.1f%% of every photo is missing from HazeFlow's output" % (100 * (1 - 1 / S)))
EOF
export PYTHONPATH=~/dehaze-eval/extra
~/mamba-venv/bin/python scripts/shoot_score.py hazeflow_a hazeflow_a+gf hazeflow_b hazeflow_b+gf 2>&1 | grep -v Warning
