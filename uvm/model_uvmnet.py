"""
Ported from Ultra/net/model.py in the real UVM-Net repo.

Output head note: the repo's README/issues mention an optional AOD-Net-style
reformulation (`output = output*x - output + 1`) as something you *could*
bolt on, but the actual shipped `model.py` doesn't use it -- it just does
`x = outc(x) + inp`, i.e. the network predicts a residual correction added
to the hazy input, and the clean image comes out directly. No separate
transmission map, no classical atmospheric-light estimate. That's a real
difference from the VSS/SS2D build already trained on this project (which
predicts t(x,y) and recovers J classically) -- this one is trained by
comparing UNet(hazy) straight against the clean ground-truth crop.

Practically: since it's direct image regression, the already-fixed
Charbonnier-loss training setup (see the collapse writeup) applies here
even more directly than it did to the transmission-map model -- just
`charbonnier(model(hazy), clean)`, no recovery formula in between. No
zero-reference/dark-channel losses needed or wanted, same lesson as before.
"""
import torch
import torch.nn as nn

from unet_parts import DoubleConv, Down, Up, OutConv


class UVMNet(nn.Module):
    def __init__(self, n_channels=3, bilinear=True, backend="torch", uvmb_size=64):
        """
        backend="torch": pure-PyTorch SSM (uvmb.py) -- no CUDA kernel needed,
            works anywhere, needed for the ONNX/Jetson export path.
        backend="cuda": real mamba_ssm.Mamba (uvmb_cuda.py) -- needs
            mamba-ssm + causal-conv1d installed on a CUDA-capable box (e.g.
            your WSL2 setup). Much faster for training; can't export to ONNX.
        uvmb_size: the fixed size UVMB always resizes to internally before
            running its SSMs (original repo hardcodes 64, i.e. 4096 tokens).
            This single number drives model3's d_model=uvmb_size**2, which
            dominates total parameter count (~695M params at the default
            64 -- see README_UVMNET.md). The repo's own README states a
            12GB GPU as the requirement; on an 8GB card, params+grads+Adam
            state alone exceed VRAM before any activation is computed.
            Lower this (e.g. 32 or 16) to fit smaller GPUs -- this is a
            deliberate capacity tradeoff, not a bug fix, so note it if you
            end up comparing results against the literal paper/repo numbers.
        """
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.backend = backend
        self.uvmb_size = uvmb_size

        self.inc = DoubleConv(n_channels, 64, backend=backend, uvmb_size=uvmb_size)
        self.down1 = Down(64, 128, backend=backend, uvmb_size=uvmb_size)
        self.down2 = Down(128, 256, backend=backend, uvmb_size=uvmb_size)
        self.down3 = Down(256, 512, backend=backend, uvmb_size=uvmb_size)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, backend=backend, uvmb_size=uvmb_size)
        self.up1 = Up(1024, 512 // factor, bilinear, backend=backend, uvmb_size=uvmb_size)
        self.up2 = Up(512, 256 // factor, bilinear, backend=backend, uvmb_size=uvmb_size)
        self.up3 = Up(256, 128 // factor, bilinear, backend=backend, uvmb_size=uvmb_size)
        self.up4 = Up(128, 64, bilinear, backend=backend, uvmb_size=uvmb_size)
        self.outc = OutConv(64, n_channels)

    def forward(self, inp):
        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x) + inp  # residual: predicts the correction, not the raw image
        return x


if __name__ == "__main__":
    # Sanity check, same shapes as the repo's own smoke test (minus the
    # hardcoded .to("cuda") -- runs on CPU here since that's what matters
    # for the pure-PyTorch scan anyway).
    data = torch.randn(2, 3, 128, 128)
    model = UVMNet(n_channels=3)
    out = model(data)
    print(out.shape)  # expect torch.Size([2, 3, 128, 128])
