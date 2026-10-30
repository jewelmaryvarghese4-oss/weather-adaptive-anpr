import cv2
import torch
from model import TransformerDehazeNet

class Defogger:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerDehazeNet(base_ch=32, bottleneck_blocks=4).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def defog(self, frame_bgr):
        """frame_bgr: raw OpenCV frame (H, W, 3) uint8, BGR."""
        h, w = frame_bgr.shape[:2]
        pad_h, pad_w = (16 - h % 16) % 16, (16 - w % 16) % 16
        padded = cv2.copyMakeBorder(frame_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            out = self.model(t)

        out = out[:, :, :h, :w]
        out_img = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        return cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)


# --- usage in your capture/detection loop ---
defogger = Defogger("checkpoints/best.pt")   # load once, outside the loop

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if weather_condition in ("fog", "haze"):     # only run when needed - saves compute
        frame = defogger.defog(frame)

    detections = yolo_model(frame)               # continue your pipeline as normal
    ...
    