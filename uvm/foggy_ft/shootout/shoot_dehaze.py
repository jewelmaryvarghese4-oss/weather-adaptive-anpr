"""Shootout stage 1: run ONE dehazer on the 50 upright photos (max side 1536) and save the outputs. One model per process, so the
code of different repositories never mixes.   python shoot_dehaze.py --model <name> [--limit N]
Models: clahe | uvm_old | uvm_ft2 | snn_l-reside-6k | snn_l-reside-ots | snn_l-reside-its | snn_l-rs-haze | fogremoval
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
H = os.path.expanduser("~/dehaze-eval")
OUT = "%s/out/%s" % (H, a.model)
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
torch.set_grad_enabled(False)
files = sorted(glob.glob(H + "/data/in1536/*.png"))
if a.limit:
    files = files[: a.limit]


def to_t(im):
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def to_pil(t):
    return Image.fromarray((t.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8))


def pad_to(x, m, mode="reflect"):
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    return torch.nn.functional.pad(x, (0, pw, 0, ph), mode=mode), h, w


# ------------------------------------------------------------------ adapters: PIL image -> PIL image
if a.model == "clahe":
    def run(im):
        lab = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

elif a.model in ("uvm_old", "uvm_ft2", "uvm_ponmudi", "uvm_gf", "uvm_pa", "uvm_gp"):
    sys.path.insert(0, "/mnt/c/projects/jewwww/uvm")
    import test_uvmnet as uvm
    from model_uvmnet import UVMNet
    ck = {"uvm_old": "checkpoints_combined/uvmnet_combined_best_stateonly.pt", "uvm_ft2": "checkpoints_rainfog2/uvmnet_combined_final.pt", "uvm_ponmudi": "checkpoints_ponmudi/uvmnet_combined_final.pt", "uvm_gf": "checkpoints_ponmudi_gf/uvmnet_combined_final.pt", "uvm_pa": "checkpoints_ponmudi_plate/uvmnet_combined_final.pt", "uvm_gp": "checkpoints_ponmudi_gopro/uvmnet_combined_final.pt"}[a.model]
    net = UVMNet(n_channels=3, backend="cuda", uvmb_size=32).to(dev)
    net.load_state_dict(torch.load("/mnt/c/projects/jewwww/uvm/" + ck, map_location=dev))
    net.eval()

    def run(im):
        return uvm.process_image(net, im, dev)

elif a.model.startswith("snn_"):
    rev = a.model[4:]
    sys.path.insert(0, H + "/repos/DehazeSNN")
    from models.hub import DehazeSNNHub
    net = DehazeSNNHub.from_pretrained("%s/weights/DehazeSNN/%s" % (H, rev)).to(dev).eval()
    print("loaded DehazeSNN", rev, "variant", net.variant, flush=True)

    def run(im):
        x = to_t(im).to(dev) * 2 - 1
        x, h, w = pad_to(x, 32)
        y = net(x).clamp(-1, 1)[..., :h, :w]
        return to_pil(y * 0.5 + 0.5)

elif a.model == "fogremoval":
    sys.path.insert(0, H + "/repos/FogRemoval")
    from networks import ResnetGenerator
    g = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=512, light=True).to(dev)
    # weights_only=True: refuse to unpickle arbitrary objects from a file that came from Dropbox
    ck = torch.load(H + "/weights/FogRemoval/NH-HAZE_params_0100000.pt", map_location=dev, weights_only=True)
    g.load_state_dict(ck["genA2B"])
    g.eval()
    T, S = 512, 384
    win = torch.hann_window(T, periodic=False).clamp(min=1e-3)
    win = (win[:, None] * win[None, :]).to(dev)

    def run(im):                                   # the generator only accepts 512x512, so the photo is processed in overlapping tiles
        x = to_t(im).to(dev) * 2 - 1
        h, w = x.shape[-2:]
        xp = torch.nn.functional.pad(x, (0, max(0, T - w), 0, max(0, T - h)), mode="reflect")
        H2, W2 = xp.shape[-2:]
        ys = sorted(set(list(range(0, H2 - T + 1, S)) + [H2 - T]))
        xs = sorted(set(list(range(0, W2 - T + 1, S)) + [W2 - T]))
        acc = torch.zeros_like(xp)
        wsum = torch.zeros(1, 1, H2, W2, device=dev)
        for y0 in ys:
            for x0 in xs:
                o = g(xp[..., y0:y0 + T, x0:x0 + T])[0]
                acc[..., y0:y0 + T, x0:x0 + T] += o * win
                wsum[..., y0:y0 + T, x0:x0 + T] += win
        return to_pil(((acc / wsum)[..., :h, :w]) * 0.5 + 0.5)
else:
    sys.exit("unknown model " + a.model)

times = []
for i, f in enumerate(files):
    im = Image.open(f).convert("RGB")
    torch.cuda.synchronize()
    t = time.time()
    out = run(im)
    torch.cuda.synchronize()
    times.append(time.time() - t)
    assert out.size == im.size, (out.size, im.size)
    out.save("%s/%s" % (OUT, os.path.basename(f)))
    if i % 10 == 0:
        print("%s %d/%d %.1fs" % (a.model, i + 1, len(files), times[-1]), flush=True)
json.dump(dict(model=a.model, photos=len(files), sec_per_photo=float(np.mean(times)), peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9),
          open(OUT + "/_timing.json", "w"))
print("DONE", a.model, "mean %.1f s per photo, peak VRAM %.2f GB" % (np.mean(times), torch.cuda.max_memory_allocated() / 1e9))
