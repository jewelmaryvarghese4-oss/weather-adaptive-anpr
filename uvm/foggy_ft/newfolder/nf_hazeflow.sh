#!/bin/bash
# HazeFlow on the new photos: raw (1.083x magnified), re-aligned with black edge (a) and edge filled from the photo (b).
export HOME=/home/jewel/nfhome
export PYTHONPATH=$HOME/dehaze-eval/shim:$HOME/dehaze-eval/extra
export TORCH_HOME=/home/jewel/.cache/torch HF_HOME=/home/jewel/.cache/huggingface
H=$HOME/dehaze-eval
PY=/home/jewel/mamba-venv/bin/python
rm -rf $H/run/HazeFlow; cp -r $H/repos/HazeFlow $H/run/HazeFlow
mkdir -p $H/run/HazeFlow/datasets/custom $H/run/HazeFlow/samples
rm -rf $H/run/HazeFlow/datasets/custom/*; cp $H/data/in1536/*.png $H/run/HazeFlow/datasets/custom/
rm -rf $H/run/HazeFlow/op
[ -f $H/run/HazeFlow/hazeflow.pth ] || cp $H/weights/HazeFlow/hazeflow.pth $H/run/HazeFlow/hazeflow.pth
cd $H/run/HazeFlow
$PY ./dehaze_sampling.py --config ./configs/rectified_flow/cifar10_rf_gaussian.py --config.data.dataset custom --config.data.test_data_root datasets/custom --config.flow.refine_t True --config.sampling.ckpt hazeflow.pth --config.work_dir samples/ --config.expr 1step --config.sampling.sample_N 1 > $H/logs/hazeflow.log 2>&1
echo "hazeflow outputs: $(ls samples/1step 2>/dev/null | wc -l)"
cd $H
$PY - <<'PY'
import glob, os
from PIL import Image
H = os.path.expanduser("~/dehaze-eval")
S = 1664 / 1536
for tag in ("hazeflow", "hazeflow_a", "hazeflow_b"):
    os.makedirs("%s/out/%s" % (H, tag), exist_ok=True)
for f in sorted(glob.glob(H + "/run/HazeFlow/samples/1step/*")):
    stem = os.path.splitext(os.path.basename(f))[0]
    ref = Image.open(H + "/data/in1536/%s.png" % stem).convert("RGB")
    o = Image.open(f).convert("RGB")
    if o.size != ref.size:
        o = o.resize(ref.size, Image.LANCZOS)
    o.save("%s/out/hazeflow/%s.png" % (H, stem))
    w, h = o.size
    back = o.resize((round(w / S), round(h / S)), Image.LANCZOS)
    for tag, bg in (("hazeflow_a", Image.new("RGB", o.size, (0, 0, 0))), ("hazeflow_b", ref.copy())):
        bg.paste(back, (0, 0)); bg.save("%s/out/%s/%s.png" % (H, tag, stem))
    print("hazeflow", stem)
PY
echo NF_HAZEFLOW_FINISHED
