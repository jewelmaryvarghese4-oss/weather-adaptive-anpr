"""
Loss terms for UVMNet training, beyond plain Charbonnier.

Motivation (from the conversation's diagnosis): the traffic-trained
checkpoint reproduces ground truth almost exactly on domain-matched data
(val PSNR ~28dB, brightness ratio ~0.98) but collapses to ~35-46% of
original brightness on real-world / off-domain photos. Charbonnier alone
gives the network no reason to keep its correction conservative outside
the training distribution -- UVMNet predicts `output = correction + input`
with the correction completely unbounded, so on unfamiliar input it can
(and does) learn/emit an arbitrarily large correction.

GradientLoss targets a separate problem (blur/detail loss from pure
pixel-wise averaging); ResidualRegularizer targets the brightness
collapse specifically by penalizing correction magnitude during training,
so the network is discouraged from relying on large corrections even when
it could reduce in-domain Charbonnier loss slightly by using one.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


class GradientLoss(nn.Module):
    """L1 distance between Sobel gradient magnitudes of pred and target.
    Penalizes blurring/detail loss that pixel-wise Charbonnier tends to
    average away, without needing a pretrained perceptual network."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = sobel_x.t()
        self.register_buffer("kx", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("ky", sobel_y.view(1, 1, 3, 3))

    def _grad_mag(self, img):
        b, c, h, w = img.shape
        img = img.reshape(b * c, 1, h, w)
        gx = F.conv2d(img, self.kx, padding=1)
        gy = F.conv2d(img, self.ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return mag.reshape(b, c, h, w)

    def forward(self, pred, target):
        return (self._grad_mag(pred) - self._grad_mag(target)).abs().mean()


class ResidualRegularizer(nn.Module):
    """Penalizes ||pred - input||^2 -- the magnitude of UVMNet's additive
    correction. Doesn't require any model change: since the model computes
    output = correction + input, the correction is recoverable exactly as
    pred - input after the fact."""

    def forward(self, pred, inp):
        return ((pred - inp) ** 2).mean()


class CombinedDehazeLoss(nn.Module):
    def __init__(self, lambda_grad=0.15, lambda_reg=0.02, eps=1e-3):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=eps)
        self.gradient = GradientLoss()
        self.reg = ResidualRegularizer()
        self.lambda_grad = lambda_grad
        self.lambda_reg = lambda_reg

    def forward(self, pred, target, inp):
        l_pix = self.charbonnier(pred, target)
        l_grad = self.gradient(pred, target)
        l_reg = self.reg(pred, inp)
        total = l_pix + self.lambda_grad * l_grad + self.lambda_reg * l_reg
        parts = {
            "charbonnier": l_pix.detach(),
            "grad": l_grad.detach(),
            "reg": l_reg.detach(),
        }
        return total, parts
