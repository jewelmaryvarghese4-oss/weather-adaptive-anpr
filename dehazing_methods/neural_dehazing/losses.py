"""
losses.py

Combined loss for dehazing training. Pure L1/L2 alone tends to produce
blurry, over-smoothed outputs on this task, so we combine:

    total = w_l1 * L1  +  w_ssim * (1 - SSIM)  +  w_perc * VGG perceptual

This mirrors the loss recipes used by AOD-Net / FFA-Net / DehazeFormer-style
papers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SSIMLoss(nn.Module):
    """Simplified single-scale SSIM, computed with a Gaussian window."""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(1)
        window = g @ g.t()
        self.register_buffer("window", window.unsqueeze(0).unsqueeze(0))

    def forward(self, img1, img2):
        C = img1.shape[1]
        window = self.window.expand(C, 1, self.window_size, self.window_size).to(img1.dtype)
        pad = self.window_size // 2

        mu1 = F.conv2d(img1, window, padding=pad, groups=C)
        mu2 = F.conv2d(img2, window, padding=pad, groups=C)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return 1 - ssim_map.mean()


class VGGPerceptualLoss(nn.Module):
    """Feature-space L1 loss using an ImageNet-pretrained VGG16 (frozen)."""

    def __init__(self, layers=(3, 8, 15)):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        prev = 0
        for l in layers:
            self.slices.append(vgg[prev:l + 1])
            prev = l + 1
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        x, y = pred, target
        for slice_ in self.slices:
            x = slice_(x)
            y = slice_(y)
            loss = loss + F.l1_loss(x, y)
        return loss


class DehazeLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.5, w_perc=0.1, use_perceptual=True):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_perc = w_perc
        self.ssim = SSIMLoss()
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual = VGGPerceptualLoss()

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim = self.ssim(pred, target)
        total = self.w_l1 * l1 + self.w_ssim * ssim
        logs = {"l1": l1.item(), "ssim_loss": ssim.item()}

        if self.use_perceptual:
            perc = self.perceptual(pred, target)
            total = total + self.w_perc * perc
            logs["perceptual"] = perc.item()

        logs["total"] = total.item()
        return total, logs