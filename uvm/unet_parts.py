"""
Ported from Ultra/net/unet_part.py in the real UVM-Net repo. Structure is
unchanged from their code.

Worth knowing (this is their design, not a guess): DoubleConv always
resizes its input down to a FIXED 64x64 before handing it to UVMB, then
resizes the UVMB output back up to whatever the actual spatial size is at
that U-Net depth, and adds it back as a residual -- only THEN does the
standard Conv-BN-ReLU-Conv-BN-ReLU run. That fixed 64x64 target is why
UVMB's channel-scan branch (model3, d_model=w*h) can have a fixed
d_model=4096 regardless of whether it's being called at the 128x128 level
or the 8x8 level deep in the encoder.

Backend switch: `backend="torch"` (default) uses the pure-PyTorch UVMB
(uvmb.py) -- no CUDA kernel needed, runs anywhere, needed for the eventual
ONNX/Jetson export path. `backend="cuda"` uses the real `mamba_ssm.Mamba`
(uvmb_cuda.py) -- much faster, but only importable on a CUDA-capable
Linux/WSL2 box with `mamba-ssm` + `causal-conv1d` installed. Use "cuda" for
training runs, "torch" for anything that has to run without that
dependency. `use_checkpoint` only applies to the torch backend -- the CUDA
kernel doesn't need it (that's the whole point of the fused kernel).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_uvmb(c, backend, use_checkpoint, size):
    if backend == "cuda":
        from uvmb_cuda import UVMBCuda
        return UVMBCuda(c=c, w=size, h=size)
    elif backend == "torch":
        from uvmb import UVMB
        return UVMB(c=c, w=size, h=size, use_checkpoint=use_checkpoint)
    else:
        raise ValueError(f"backend must be 'torch' or 'cuda', got {backend!r}")


class DoubleConv(nn.Module):
    """UVMB (at a fixed size x size) + residual, then (conv => BN => ReLU) * 2.

    `uvmb_size` is the internal resize target UVMB always runs its SSMs at
    (the original repo hardcodes 64, i.e. 4096 tokens -- see README_UVMNET.md's
    memory finding). That single number drives model3's d_model=size*size,
    which dominates total parameter count. Shrinking it is the real lever
    for fitting this on a smaller GPU: going 64->32 cuts model3's Linear
    layers ~4x in each dimension (~16x fewer params in that layer alone).
    """

    def __init__(self, in_channels, out_channels, mid_channels=None,
                 use_checkpoint=True, backend="torch", uvmb_size=64):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.uvmb_size = uvmb_size
        self.ub = _make_uvmb(in_channels, backend, use_checkpoint, uvmb_size)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        inputs = F.interpolate(x, size=[self.uvmb_size, self.uvmb_size],
                                mode='bilinear', align_corners=True)
        outputs = self.ub(inputs)
        outputs = F.interpolate(
            outputs, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=True
        ) + x
        return self.double_conv(outputs)


class Down(nn.Module):
    """Downscaling with maxpool then DoubleConv."""

    def __init__(self, in_channels, out_channels, use_checkpoint=True, backend="torch", uvmb_size=64):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_checkpoint=use_checkpoint,
                       backend=backend, uvmb_size=uvmb_size),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv, with skip connection."""

    def __init__(self, in_channels, out_channels, bilinear=True, use_checkpoint=True,
                 backend="torch", uvmb_size=64):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2,
                                    use_checkpoint=use_checkpoint, backend=backend, uvmb_size=uvmb_size)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_checkpoint=use_checkpoint,
                                    backend=backend, uvmb_size=uvmb_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
