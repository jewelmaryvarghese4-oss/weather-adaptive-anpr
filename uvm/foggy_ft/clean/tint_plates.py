"""Plate check for the tint fix: out/uvm_gp (network at 1536 px) -> guided filter r4 -> tint fix, saved as out/uvm_gpt (already filtered; score it WITHOUT +gf)."""
import glob, os, sys
import cv2, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); UVM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, UVM); sys.path.insert(0, HERE)
from gf_head import guided_filter_head
from tint_fix import fix_tint
H = os.path.expanduser("~/dehaze-eval"); OUT = H + "/out/uvm_gpt"; os.makedirs(OUT, exist_ok=True)
t = lambda x: torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255
n = 0
for f in sorted(glob.glob(H + "/out/uvm_gp/*.png")):
    st = os.path.basename(f)
    hz = cv2.imread(H + "/data/in1536/" + st); o = cv2.imread(f)
    if hz is None or o is None or hz.shape != o.shape: continue
    g = guided_filter_head(t(o), t(hz), 4, 1e-2)[0].permute(1, 2, 0).clamp(0, 1).numpy()
    g = cv2.cvtColor((g * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUT + "/" + st, fix_tint(hz, g, 0.8)); n += 1
print("tint-fixed 1536 px outputs:", n)
