"""
losses.py — supervised version, using your paired clear/hazy data
instead of zero-reference priors.

Why this instead of the zero-reference patch: your dataset has a real
`clear/` ground-truth image for every `hazy/` variant (crop_XXXX.jpg <->
crop_XXXX_v0..v7.jpg). That means the collapse-to-t_min failure mode
(diagnosed from your training run: t was locked at exactly 0.100 for
every pixel of every image) can be fixed at the root instead of patched
around: supervise the network's final output DIRECTLY against the real
clean image. Collapsing t to a constant now produces a huge reconstruction
error against ground truth, so there's no "free win" left for the
optimizer the way there was with dark_channel_loss alone.

Charbonnier loss = a smoothed, more robust version of L1:
    L(x, y) = sqrt((x - y)^2 + eps^2)
It's the standard reconstruction loss in image restoration/super-resolution
work (used in e.g. LapSRN, MPRNet) — behaves like L2 near zero (so it's
differentiable/well-behaved at x=y, unlike raw L1) and like L1 far from
zero (so it's less sensitive to outlier pixels/big errors than raw L2,
i.e. doesn't let a few bad pixels dominate the gradient the way MSE does).

Exposure loss is kept as an auxiliary term (not the main signal anymore —
Charbonnier against ground truth already anchors brightness correctly,
this just gives a small extra nudge/regularizer and costs nothing to keep).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier_loss(pred, target, eps: float = 1e-3):
    """pred, target: (B,3,H,W) in [0,1], same shape."""
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


def exposure_loss(output, target_level: float = 0.6, patch_size: int = 16):
    pooled = F.avg_pool2d(output.mean(dim=1, keepdim=True), patch_size)
    return ((pooled - target_level) ** 2).mean()


class DehazeLoss(nn.Module):
    """
    Combined supervised loss: Charbonnier (primary, against real ground
    truth) + exposure control (auxiliary regularizer).

    forward() signature: (output, clear_gt) — note this is DIFFERENT
    from both your original unsupervised version (output, transmission)
    and my earlier zero-reference patch (output, transmission, hazy).
    This version needs the paired clean ground-truth image instead of
    the transmission map, since supervision no longer routes through
    the zero-reference priors at all.
    """

    def __init__(self, w_charbonnier: float = 1.0, w_exposure: float = 0.1,
                 charbonnier_eps: float = 1e-3, target_exposure: float = 0.6):
        super().__init__()
        self.w_charbonnier = w_charbonnier
        self.w_exposure = w_exposure
        self.charbonnier_eps = charbonnier_eps
        self.target_exposure = target_exposure

    def forward(self, output: torch.Tensor, clear_gt: torch.Tensor):
        l_char = charbonnier_loss(output, clear_gt, eps=self.charbonnier_eps)
        l_exp = exposure_loss(output, self.target_exposure)

        total = self.w_charbonnier * l_char + self.w_exposure * l_exp

        logs = {
            "charbonnier": l_char.item(),
            "exposure": l_exp.item(),
            "total": total.item(),
        }
        return total, logs
