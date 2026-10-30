set -e
H=~/dehaze-eval
~/mamba-venv/bin/pip install --quiet --target $H/extra --no-deps absl-py ml_collections six contextlib2 2>&1 | tail -2
mkdir -p $H/shim/op $H/shim/pyiqa
# replacement for HazeFlow's compiled CUDA op: the repo's own pure-PyTorch version of upfirdn2d (same maths, no compiling)
python3 - <<'EOF'
import os, re
H = os.path.expanduser("~/dehaze-eval")
src = open(H + "/repos/HazeFlow/op/upfirdn2d.py").read()
native = src[src.index("def upfirdn2d_native("):]
open(H + "/shim/op/__init__.py", "w").write(
    "import torch\nimport torch.nn.functional as F\n\n\n"
    "def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):\n"
    "    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])\n\n\n" + native + "\n")
open(H + "/shim/pyiqa/__init__.py", "w").write(
    "import torch\n\n\ndef create_metric(*a, **k):\n    return lambda x: torch.zeros(1)   # image-quality metrics are not needed here\n")
print("shims written")
EOF
rm -rf $H/run/HazeFlow; mkdir -p $H/run; cp -r $H/repos/HazeFlow $H/run/HazeFlow
mkdir -p $H/run/HazeFlow/datasets/custom $H/run/HazeFlow/samples
cp $H/data/in1536/*.png $H/run/HazeFlow/datasets/custom/
rm -rf $H/run/HazeFlow/op
echo "prepared: $(ls $H/run/HazeFlow/datasets/custom | wc -l) photos; weight file: $(ls -la $H/weights/HazeFlow | tail -1 | awk '{print $5, $9}')"
