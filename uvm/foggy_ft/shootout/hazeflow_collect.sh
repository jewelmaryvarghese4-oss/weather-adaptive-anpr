cd ~/dehaze-eval
ls run/HazeFlow/samples/1step | head -3
~/mamba-venv/bin/python - <<'EOF'
import glob, os, shutil
from PIL import Image
H = os.path.expanduser("~/dehaze-eval")
src = sorted(glob.glob(H + "/run/HazeFlow/samples/1step/*"))
os.makedirs(H + "/out/hazeflow", exist_ok=True)
bad = 0
for f in src:
    stem = os.path.splitext(os.path.basename(f))[0]
    ref = H + "/data/in1536/%s.png" % stem
    if not os.path.exists(ref):
        print("no matching input for", os.path.basename(f)); bad += 1; continue
    im = Image.open(f).convert("RGB")
    r = Image.open(ref)
    if im.size != r.size:
        print("size differs", stem, im.size, r.size)
        im = im.resize(r.size, Image.LANCZOS)
        bad += 1
    im.save(H + "/out/hazeflow/%s.png" % stem)
print("collected", len(os.listdir(H + "/out/hazeflow")), "outputs; problems:", bad)
EOF
export PYTHONPATH=~/dehaze-eval/extra
~/mamba-venv/bin/python scripts/shoot_score.py hazeflow hazeflow+gf 2>&1 | grep -v Warning
