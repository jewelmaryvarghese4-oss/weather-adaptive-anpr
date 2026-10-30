"""
model.py
--------
MambaDehazeNet: a small VSS (Vision-Mamba) U-Net that predicts an
atmospheric-scattering transmission map t(x,y), used to recover a clear
image WITHOUT needing paired clean/hazy ground truth.

Atmospheric Scattering Model (Koschmieder / He et al. dark-channel prior):
    I(x) = J(x) * t(x) + A * (1 - t(x))
  where:
    I = observed hazy image
    J = the clear scene we want to recover
    A = global atmospheric light (scalar per channel, estimated classically)
    t = transmission map in [0,1]; t=1 means no haze, t->0 means dense haze

  Rearranged to recover J once t and A are known:
    J(x) = (I(x) - A) / max(t(x), t_min) + A

The network's job is only to predict t(x,y) (a single-channel map the same
size as the image). A is estimated with a classical dark-channel heuristic
(no learning needed, very cheap). Because we reconstruct J analytically
from I, t and A, we can train the network with *unsupervised* priors
(dark-channel loss, TV smoothness, color constancy) instead of needing
paired clean images — see losses.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_blocks import VSSBlock


def to_bhwc(x):
    return x.permute(0, 2, 3, 1)


def to_bchw(x):
    return x.permute(0, 3, 1, 2)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=32, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)  # (B, C, H/p, W/p)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)


class VSSStage(nn.Module):
    """A few stacked VSS blocks operating at a fixed resolution/channel width."""

    def __init__(self, dim, depth=2, d_state=16):
        super().__init__()
        self.blocks = nn.ModuleList([VSSBlock(dim, d_state=d_state) for _ in range(depth)])

    def forward(self, x_bchw):
        x = to_bhwc(x_bchw)
        for blk in self.blocks:
            x = blk(x)
        return to_bchw(x)


class MambaDehazeNet(nn.Module):
    """
    Small U-Net with VSS (Mamba) stages instead of plain conv blocks.
    Input:  hazy RGB image   (B, 3, H, W), H and W should be divisible by 8
    Output: transmission map (B, 1, H, W) in (t_min, 1)
    """

    def __init__(self, embed_dim=32, depths=(2, 2, 2), d_state=16, t_min=0.1):
        super().__init__()
        self.t_min = t_min
        C = embed_dim

        self.patch_embed = PatchEmbed(3, C, patch_size=2)  # H/2

        self.enc1 = VSSStage(C, depths[0], d_state)         # H/2
        self.down1 = Downsample(C)                          # H/4, C*2
        self.enc2 = VSSStage(C * 2, depths[1], d_state)      # H/4
        self.down2 = Downsample(C * 2)                       # H/8, C*4

        self.bottleneck = VSSStage(C * 4, depths[2], d_state)  # H/8

        self.up2 = Upsample(C * 4)                           # H/4, C*2
        self.fuse2 = nn.Conv2d(C * 4, C * 2, 1)               # after concat skip
        self.dec2 = VSSStage(C * 2, depths[1], d_state)

        self.up1 = Upsample(C * 2)                            # H/2, C
        self.fuse1 = nn.Conv2d(C * 2, C, 1)
        self.dec1 = VSSStage(C, depths[0], d_state)

        self.to_pixels = nn.ConvTranspose2d(C, C, kernel_size=2, stride=2)  # back to H
        self.head = nn.Conv2d(C, 1, kernel_size=3, padding=1)

    def forward(self, x):
        B, _, H, W = x.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        e0 = self.patch_embed(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))

        d2 = self.up2(b)
        d2 = self.fuse2(torch.cat([d2, e2], dim=1))
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.fuse1(torch.cat([d1, e1], dim=1))
        d1 = self.dec1(d1)

        out = self.to_pixels(d1)
        t = torch.sigmoid(self.head(out))
        t = self.t_min + t * (1.0 - self.t_min)  # rescale to (t_min, 1)

        if pad_h or pad_w:
            t = t[:, :, :H, :W]
        return t


@torch.no_grad()
def estimate_atmospheric_light(img, dark_channel_percentile=0.001):
    """
    Classical dark-channel-prior estimate of global atmospheric light A.
    img: (B, 3, H, W) in [0,1]
    Returns A: (B, 3, 1, 1)
    """
    B, C, H, W = img.shape
    dark = img.min(dim=1, keepdim=True).values  # crude single-scale dark channel
    # min-filter (erosion) with a small window approximates the classic patch-wise dark channel
    dark = -F.max_pool2d(-dark, kernel_size=7, stride=1, padding=3)

    n_pixels = H * W
    n_top = max(1, int(n_pixels * dark_channel_percentile))
    flat_dark = dark.reshape(B, -1)
    _, idx = torch.topk(flat_dark, n_top, dim=1)

    img_flat = img.reshape(B, C, -1)
    A = torch.zeros(B, C, 1, 1, device=img.device)
    for b in range(B):
        brightest = img_flat[b, :, idx[b]]  # (C, n_top)
        A[b, :, 0, 0] = brightest.max(dim=1).values
    return A


def recover_clear_image(hazy, t, A, t_min=0.1):
    """J(x) = (I(x) - A) / max(t(x), t_min) + A"""
    t_clamped = torch.clamp(t, min=t_min)
    J = (hazy - A) / t_clamped + A
    return torch.clamp(J, 0.0, 1.0)