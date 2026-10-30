"""
Training skeleton for UVMNet (the faithful UVM-Net port).

Deliberately supervised (Charbonnier against real clean/hazy pairs), NOT
zero-reference/dark-channel priors -- that's the exact fix that got the
VSS/SS2D model out of its collapse (t -> 0.1 everywhere, dark_channel_loss's
degenerate minimizer). UVMNet predicts the clean image directly (residual
regression, see model_uvmnet.py's docstring), so there's no recovery
formula in between: loss is just charbonnier(model(hazy), clean).

Fill in your actual dataset loader -- this assumes the same layout already
in use (datasets/dataset/{clear,hazy}/, paired filenames).
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model_uvmnet import UVMNet


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


class PairedHazeDataset(Dataset):
    """clear/<name>.png <-> hazy/<name>_<variant>.png, adjust to your actual naming."""

    def __init__(self, root, crop_size=64):
        self.clear_dir = os.path.join(root, "clear")
        self.hazy_dir = os.path.join(root, "hazy")
        self.pairs = self._index_pairs()
        self.tf = transforms.Compose([
            # Native crops are small and non-square (matches the project's
            # established ~25x139-ish plate-crop scale) -- smaller than any
            # sensible training crop size, so RandomCrop can't apply here.
            # Resize (not crop) to a fixed square size instead; this matches
            # what UVMB does internally anyway (interpolates to 64x64 at
            # every U-Net stage regardless of native input size).
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
        ])

    def _index_pairs(self):
        # Matches the real naming convention already in use for this project:
        # hazy = "<name>_v<N>.<ext>" (e.g. crop_0000_v3.jpg), clear = "<name>.<ext>"
        import re
        hazy_files = sorted(os.listdir(self.hazy_dir))
        pairs = []
        for hf in hazy_files:
            stem, ext = os.path.splitext(hf)
            base = re.sub(r"_v\d+$", "", stem)
            clear_candidate = base + ext
            if os.path.exists(os.path.join(self.clear_dir, clear_candidate)):
                pairs.append((hf, clear_candidate))
        if not pairs:
            raise RuntimeError(
                f"No hazy/clear pairs matched in {self.hazy_dir} / {self.clear_dir}. "
                "Check the naming convention against _index_pairs() -- adjust the "
                "regex if your actual filenames differ from '<name>_v<N>.<ext>'."
            )
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_name, clear_name = self.pairs[idx]
        hazy = Image.open(os.path.join(self.hazy_dir, hazy_name)).convert("RGB")
        clear = Image.open(os.path.join(self.clear_dir, clear_name)).convert("RGB")
        # Resize is deterministic (unlike the RandomCrop this replaced), so no
        # need to sync a random seed between the two anymore.
        hazy_t = self.tf(hazy)
        clear_t = self.tf(clear)
        return hazy_t, clear_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="datasets/dataset")
    ap.add_argument("--crop_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default="cuda" if torch.cuda.is_available() else "torch",
                     choices=["torch", "cuda"],
                     help="'cuda' = real mamba_ssm (needs mamba-ssm + causal-conv1d installed, "
                          "e.g. your WSL2 setup). 'torch' = pure-PyTorch, no CUDA kernel, "
                          "works anywhere but much slower.")
    ap.add_argument("--ckpt_out", default="uvmnet_ckpt.pt")
    ap.add_argument("--uvmb_size", type=int, default=64,
                     help="Internal SSM resize target (original repo default: 64, "
                          "~695M params, needs ~12GB GPU per the repo's own README). "
                          "Lower this (e.g. 32 or 16) to fit smaller VRAM -- see "
                          "model_uvmnet.py's docstring for the tradeoff.")
    args = ap.parse_args()

    if args.device == "cpu":
        # See README_UVMNET.md: a single UVMB block at c=512, with grad,
        # took ~70s fwd+backward on this test machine even WITH
        # checkpointing. The full 9-block network at batch>=1 on CPU is
        # not a realistic training setup -- this flag exists so you don't
        # discover that an hour into a run. Use a GPU for actual training,
        # same requirement the original repo states.
        print("WARNING: training UVMNet on CPU will be extremely slow "
              "(a single deep UVMB block took ~70s fwd+backward in "
              "testing). Strongly recommend a CUDA device for training.")

    dataset = PairedHazeDataset(args.data_root, crop_size=args.crop_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = UVMNet(n_channels=3, backend=args.backend, uvmb_size=args.uvmb_size).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    charbonnier = CharbonnierLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for hazy, clear in loader:
            hazy, clear = hazy.to(args.device), clear.to(args.device)
            pred = model(hazy)
            loss = charbonnier(pred, clear)

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        print(f"epoch {epoch:3d} | charbonnier {running / len(loader):.4f} | lr {sched.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), args.ckpt_out)
    print(f"saved -> {args.ckpt_out}")


if __name__ == "__main__":
    main()
