"""
Loss for (hazy, clear) pairs that show the same place but are NOT pixel-aligned. No pixel-wise comparison between the output and the clear frame;
only what survives a few metres of camera offset:
  low   : luminance on a coarse 8x8 grid (sky/road/vegetation tone layout) -- tolerant to shifts of a cell
  tone  : luminance mean/std of the crop
  dark  : dark-channel level (fog lifts it; clear scenes have a low one)
  edge  : local contrast must not stay below the clear frame's (hinge on log gradient energy)
  struct: the output keeps the hazy input's edge directions (so nothing is invented)
Colour is deliberately left to the aligned datasets: the clear passes are all grey overcast roads.
"""
import torch
import torch.nn.functional as F


def _lum(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def _grad(y):
    gx = y[:, :, :, 1:] - y[:, :, :, :-1]
    gy = y[:, :, 1:, :] - y[:, :, :-1, :]
    return gx[:, :, :-1, :], gy[:, :, :, :-1]


def _dark(x, k=15):
    return -F.max_pool2d(-x.min(1, keepdim=True)[0], k, stride=1, padding=k // 2)


def unaligned_loss(pred, ref, hazy, w=(1.0, 0.5, 0.5, 0.1, 0.2)):
    pred, ref, hazy = pred.float().clamp(0, 1), ref.float(), hazy.float()
    yp, yr, yh = _lum(pred), _lum(ref), _lum(hazy)
    cell = pred.shape[-1] // 8
    low = F.l1_loss(F.avg_pool2d(yp, cell), F.avg_pool2d(yr, cell))
    tone = (yp.mean((1, 2, 3)) - yr.mean((1, 2, 3))).abs().mean() + (yp.std((1, 2, 3)) - yr.std((1, 2, 3))).abs().mean()
    dark = (_dark(pred).mean((1, 2, 3)) - _dark(ref).mean((1, 2, 3))).abs().mean()
    gpx, gpy = _grad(yp)
    grx, gry = _grad(yr)
    ep = (gpx.abs().mean((1, 2, 3)) + gpy.abs().mean((1, 2, 3))) + 1e-4
    er = (grx.abs().mean((1, 2, 3)) + gry.abs().mean((1, 2, 3))) + 1e-4
    edge = F.relu(torch.log(er) - torch.log(ep)).mean()
    ghx, ghy = _grad(yh)
    mag = (ghx ** 2 + ghy ** 2).sqrt() + 1e-4
    cos = (gpx * ghx + gpy * ghy) / (((gpx ** 2 + gpy ** 2).sqrt() + 1e-4) * mag)
    struct = (1 - cos) * (mag / mag.mean((1, 2, 3), keepdim=True))
    struct = struct.mean() * 0.5
    parts = {"low": low.item(), "tone": tone.item(), "dark": dark.item(), "edge": edge.item(), "struct": struct.item()}
    return w[0] * low + w[1] * tone + w[2] * dark + w[3] * edge + w[4] * struct, parts
