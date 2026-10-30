"""
Seamless full-size dehazing without the tile seam, and with the tan cast pulled back.
 1. the photo is shrunk to max side 1536 and goes through the network as ONE tile (no tiles -> no per-tile haze/airlight estimates -> no seam;
    1536 px is also the scale the Ponmudi training matches),
 2. guided filter r4 on that result (as trained),
 3. the result is brought back to the photo's full size by JOINT UPSAMPLING: bicubic up, then a guided filter whose guide is the full-size hazy photo
    (edges and fine detail come from the photo itself),
 4. tint_fix.py removes the warm cast in dense fog.
    python foggy_ft/clean/dehaze_clean.py --input ../datasets/foggy_photos_upright --output ../datasets/foggy_photos_uvm_gopro_clean
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
UVM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, UVM)
sys.path.insert(0, HERE)
import test_uvmnet as uvm
from gf_head import guided_filter_head
from model_uvmnet import UVMNet
from tint_fix import fix_tint

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--checkpoint", default="checkpoints_ponmudi_gopro/uvmnet_combined_final.pt")
ap.add_argument("--long_side", type=int, default=1536)
ap.add_argument("--up_radius", type=int, default=8)
ap.add_argument("--tint_beta", type=float, default=0.8)
a = ap.parse_args()
os.makedirs(a.output, exist_ok=True)
dev = torch.device("cuda")
torch.set_grad_enabled(False)
net = UVMNet(n_channels=3, backend="cuda", uvmb_size=32).to(dev)
net.load_state_dict(torch.load(os.path.join(UVM, a.checkpoint), map_location=dev))
net.eval()


def tens(bgr):
    return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255.0


def gf(pred_bgr, guide_bgr, r, device="cpu"):
    o = guided_filter_head(tens(pred_bgr).to(device), tens(guide_bgr).to(device), r, 1e-2)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return cv2.cvtColor((o * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


for f in sorted(glob.glob(os.path.join(a.input, "*.jpg")) + glob.glob(os.path.join(a.input, "*.png"))):
    stem = os.path.splitext(os.path.basename(f))[0]
    hazy = cv2.imread(f)
    H, W = hazy.shape[:2]
    s = min(1.0, a.long_side / max(H, W))
    small = cv2.resize(hazy, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else hazy
    out = uvm.process_image(net, Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), dev)
    out = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
    out = gf(out, small, 4, "cuda")
    torch.cuda.empty_cache()
    if s < 1:
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)
        out = gf(out, hazy, a.up_radius)
    out = fix_tint(hazy, out, a.tint_beta)
    cv2.imwrite(os.path.join(a.output, stem + "_restored.png"), out)
    print("done", stem, flush=True)
