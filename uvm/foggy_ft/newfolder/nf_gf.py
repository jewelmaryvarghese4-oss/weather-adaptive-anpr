"""guided filter r4 (guide = the photo) over a folder of raw network outputs: nf_gf.py <hazy_dir> <raw_dir> <out_dir>"""
import glob, os, sys
import cv2, numpy as np, torch
sys.path.insert(0, "/mnt/c/projects/jewwww/uvm")
from gf_head import guided_filter_head
hz, raw, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
t = lambda x: torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255
for f in sorted(glob.glob(raw + "/*_restored.png")):
    st = os.path.basename(f).replace("_restored.png", "")
    hp = hz + "/%s.jpg" % st
    if not os.path.exists(hp):
        hp = hz + "/%s.png" % st
    h = cv2.imread(hp); o = cv2.imread(f)
    g = guided_filter_head(t(o), t(h), 4, 1e-2)[0].permute(1, 2, 0).clamp(0, 1).numpy()
    cv2.imwrite(out + "/%s_restored.png" % st, cv2.cvtColor((g * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("gf", st, flush=True)
