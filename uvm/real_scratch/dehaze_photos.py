"""
Dehaze a folder of photos with a model trained by train_uvm_real.py (or any UVMNet state dict).
Each photo is shrunk to max side --long_side, goes through the network as ONE piece (no tiles, so no seams), and is brought back to its full size by
joint upsampling (bicubic + a guided filter whose guide is the original photo), so edges and detail come from the photo itself.
Writes <stem>_restored.png (same naming as test_uvmnet.py, so the gallery scripts find them).
    python real_scratch/dehaze_photos.py --checkpoint checkpoints_real_scratch/real_final_ema.pt --input ../datasets/foggy_photos_upright --output ../datasets/foggy_photos_uvm_real
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
UVM = os.path.dirname(HERE)
sys.path.insert(0, UVM)
import test_uvmnet as uvm
from gf_head import guided_filter_head
from model_uvmnet import UVMNet

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--long_side", type=int, default=1536)
ap.add_argument("--up_radius", type=int, default=8)
ap.add_argument("--uvmb_size", type=int, default=32)
a = ap.parse_args()
os.makedirs(a.output, exist_ok=True)
dev = torch.device("cuda")
torch.set_grad_enabled(False)
net = UVMNet(n_channels=3, backend="cuda", uvmb_size=a.uvmb_size).to(dev)
net.load_state_dict(torch.load(a.checkpoint, map_location=dev))
net.eval()
t = lambda x: torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255.0
files = sorted(glob.glob(os.path.join(a.input, "*.jpg")) + glob.glob(os.path.join(a.input, "*.png")) + glob.glob(os.path.join(a.input, "*.jpeg")))
for f in tqdm(files, desc="dehazing", unit="photo", dynamic_ncols=True):
    stem = os.path.splitext(os.path.basename(f))[0]
    hazy = cv2.imread(f)
    H, W = hazy.shape[:2]
    s = min(1.0, a.long_side / max(H, W))
    small = cv2.resize(hazy, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else hazy
    out = uvm.process_image(net, Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), dev)
    out = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
    torch.cuda.empty_cache()
    if s < 1:
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)
        g = guided_filter_head(t(out), t(hazy), a.up_radius, 1e-2)[0].permute(1, 2, 0).clamp(0, 1).numpy()
        out = cv2.cvtColor((g * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(a.output, stem + "_restored.png"), out)
print("images:", len(files), "->", a.output)
