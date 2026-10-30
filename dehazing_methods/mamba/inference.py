"""
inference.py
------------
Run the trained Mamba dehazing model on foggy images, and a helper
function `dehaze_frame()` meant to be dropped directly into stage 6
("Weather enhancement") of your ANPR pipeline:

    Camera -> Weather analysis -> Vehicle detection -> Plate detection ->
    Rain/Fog enhancement -> Best-frame selection -> Perspective correction
    -> OCR -> Verification

Usage (single image or a folder):
    python inference.py --checkpoint checkpoints/mamba_dehaze_epoch50.pt \
        --input /path/to/foggy_or_folder --output ./dehazed_out
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from model import MambaDehazeNet, estimate_atmospheric_light, recover_clear_image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MambaDehazeNet(embed_dim=ckpt.get("embed_dim", 32),
                            t_min=ckpt.get("t_min", 0.1)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("t_min", 0.1)


@torch.no_grad()
def dehaze_frame(model, frame_bgr_or_rgb, t_min=0.1, device="cpu", is_bgr=False):
    """
    Drop-in function for the ANPR pipeline's weather-enhancement stage.

    Args:
        model: loaded MambaDehazeNet (eval mode)
        frame_bgr_or_rgb: HxWx3 uint8 numpy array (e.g. straight from
            OpenCV `cv2.VideoCapture` / `cv2.imread`, which is BGR)
        is_bgr: set True if the frame came from OpenCV (BGR channel order)

    Returns:
        HxWx3 uint8 numpy array, same channel order as the input, dehazed.
    """
    frame = frame_bgr_or_rgb
    if is_bgr:
        frame = frame[:, :, ::-1]  # BGR -> RGB

    tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device)

    A = estimate_atmospheric_light(tensor)
    t = model(tensor)
    output = recover_clear_image(tensor, t, A, t_min=t_min)

    out_np = (output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    if is_bgr:
        out_np = out_np[:, :, ::-1]  # RGB -> BGR
    return out_np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="Image file or folder")
    p.add_argument("--output", type=str, default="./dehazed_out")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device)
    model, t_min = load_model(args.checkpoint, device)

    if os.path.isdir(args.input):
        paths = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
                  if f.lower().endswith(IMG_EXTS)]
    else:
        paths = [args.input]

    to_tensor = T.ToTensor()
    for path in paths:
        img = Image.open(path).convert("RGB")
        arr = np.array(img)  # RGB, HxWx3, uint8
        out = dehaze_frame(model, arr, t_min=t_min, device=device, is_bgr=False)
        out_img = Image.fromarray(out)
        out_path = os.path.join(args.output, os.path.basename(path))
        out_img.save(out_path)
        print(f"dehazed -> {out_path}")


if __name__ == "__main__":
    main()