"""
Loss for training a dehazer on REAL fog images only, where no pixel-aligned clear partner exists.

Two kinds of training samples:
  paired   (GoProHazy): the clear reference shows the same place on the road, taken a few metres to the side -> layout term + statistics
  unpaired (your Ponmudi fog photos / video frames): the reference is any real clear image -> batch-level statistics only
Every term is a statistic that survives that kind of mismatch; nothing compares output and reference pixel by pixel.

  low    luminance on a coarse 8x8 grid (paired only): sky / road / vegetation tone layout
  tone   luminance mean and std (per sample if paired, batch mean if unpaired)
  dark   dark-channel level: fog lifts it, clear scenes have a low one
  edge   local contrast may not stay below the clear reference's (hinge on log gradient energy)
  struct the output keeps the hazy input's edge directions (nothing is invented)
  hue    low-frequency colour direction stays the input's (fog must not change the hue of things)
  sat    dehazing must not desaturate (hinge on low-frequency saturation)
  noise  finest-scale energy (grain) may not exceed that of real clear images
  affine the output must be explainable as a smooth local gain+offset of the hazy input (no invented texture, no halos, no colour blow-ups)
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


def _stats(x):
    """per-sample statistics: luminance mean, std, dark-channel mean, gradient energy"""
    y = _lum(x)
    gx, gy = _grad(F.avg_pool2d(y, 3, stride=1, padding=1))      # a little smoothing: grain must not count as contrast
    fine = (y - F.avg_pool2d(y, 3, stride=1, padding=1)).abs().mean((1, 2, 3)) + 1e-4        # finest-scale energy = grain
    return torch.stack([y.mean((1, 2, 3)), y.std((1, 2, 3)), _dark(x).mean((1, 2, 3)), gx.abs().mean((1, 2, 3)) + gy.abs().mean((1, 2, 3)) + 1e-4, fine], 1)


def _box(x, r):
    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


def _affine_residual(pred, hazy, r=8, eps=1e-3):
    """Dehazing is a smooth local gain + offset of the hazy image (airlight and transmission change slowly). Fit out = a*in + b in every window of the
    image (per colour channel, like a guided filter) and measure what that cannot explain: invented texture, halos around edges, colour blow-ups."""
    mi, mo = _box(hazy, r), _box(pred, r)
    a = (_box(hazy * pred, r) - mi * mo) / (_box(hazy * hazy, r) - mi * mi + eps)
    b = mo - a * mi
    return (pred - (_box(a, r) * hazy + _box(b, r))).abs().mean()


def sat(x):
    return (x.max(1, keepdim=True)[0] - x.min(1, keepdim=True)[0]) / (x.max(1, keepdim=True)[0] + 1e-3)


def _chroma(x, cell):
    a = F.avg_pool2d(x[:, 0:1] - x[:, 1:2], cell)
    b = F.avg_pool2d(0.5 * (x[:, 0:1] + x[:, 1:2]) - x[:, 2:3], cell)
    return a, b


WEIGHTS = dict(low=1.0, tone=0.5, dark=0.5, edge=0.2, struct=0.4, hue=0.5, sat=0.3, noise=0.6, affine=0.5, chroma_cap=4.0, chroma_ref=0.0, sat_cap=1.5, dcp=0.0)


def real_loss(pred, ref, hazy, paired, w=WEIGHTS):
    """pred/ref/hazy (B,3,H,W) in [0,1]; paired (B,) bool tensor. Returns (loss, parts dict)."""
    pred, ref, hazy = pred.float().clamp(0, 1), ref.float(), hazy.float()
    paired = paired.bool()
    unp = ~paired
    cell = pred.shape[-1] // 8
    yp, yr, yh = _lum(pred), _lum(ref), _lum(hazy)
    zero = pred.sum() * 0
    # layout (paired only)
    low = F.l1_loss(F.avg_pool2d(yp[paired], cell), F.avg_pool2d(yr[paired], cell)) if paired.any() else zero
    # statistics: per sample if paired, batch mean if unpaired
    sp, sr = _stats(pred), _stats(ref)
    parts_stat = []
    for cols, name in (([0, 1], "tone"), ([2], "dark")):
        t = zero
        if paired.any():
            t = t + (sp[paired][:, cols] - sr[paired][:, cols]).abs().mean()
        if unp.any():
            t = t + (sp[unp][:, cols].mean(0) - sr[unp][:, cols].mean(0)).abs().mean()
        parts_stat.append(t)
    tone, dark = parts_stat
    edge = zero
    if paired.any():
        d = torch.log(sr[paired][:, 3]) - torch.log(sp[paired][:, 3])
        edge = edge + (F.relu(d) + 0.5 * F.relu(-d)).mean()          # too little contrast is penalised fully, too much (halos, grain) at half weight
    if unp.any():
        d = torch.log(sr[unp][:, 3].mean()) - torch.log(sp[unp][:, 3].mean())
        edge = edge + F.relu(d) + 0.5 * F.relu(-d)
    # grain: the output may not have more finest-scale energy than real clear images do (dehazing multiplies contrast, and with it the grain)
    noise = zero
    if paired.any():
        noise = noise + F.relu(torch.log(sp[paired][:, 4]) - torch.log(sr[paired][:, 4])).mean()
    if unp.any():
        noise = noise + F.relu(torch.log(sp[unp][:, 4].mean()) - torch.log(sr[unp][:, 4].mean()))
    affine = _affine_residual(pred, hazy)
    ao, bo = _chroma(pred, 16)
    ai, bi = _chroma(hazy, 16)
    ci = (ai ** 2 + bi ** 2 + 1e-8).sqrt()
    co = (ao ** 2 + bo ** 2 + 1e-8).sqrt()
    # colour cast (the hue term is blind where the input is grey fog, so the colour was free there and drifted green):
    #  chroma_cap : low-frequency colourfulness may not exceed 3x the input's (+ a small slack) -- fog stays neutral, objects may gain colour
    #  chroma_ref : mean colour direction of the output matches real clear images (per sample if paired, batch mean if unpaired)
    #  sat_cap    : mean saturation may not exceed the real clear images' by more than 10%
    chroma_cap = F.relu(co - (3.0 * ci + 0.02)).mean()
    cm = lambda x: torch.cat(_chroma(x, 16), 1).mean((2, 3))                  # (B, 2) mean (R-G, (R+G)/2-B)
    mo_, mr_ = cm(pred), cm(ref)
    chroma_ref = zero
    if paired.any():
        chroma_ref = chroma_ref + (mo_[paired] - mr_[paired]).abs().mean()
    if unp.any():
        chroma_ref = chroma_ref + (mo_[unp].mean(0) - mr_[unp].mean(0)).abs().mean()
    sm_ = lambda x: sat(x).mean((1, 2, 3))
    so, sr_ = sm_(pred), sm_(ref)
    sat_cap = zero
    if paired.any():
        sat_cap = sat_cap + F.relu(so[paired] - 1.1 * sr_[paired]).mean()
    if unp.any():
        sat_cap = sat_cap + F.relu(so[unp].mean() - 1.1 * sr_[unp].mean())
    # fog removal, photo by photo (unpaired = your real Ponmudi fog): a clear photo has a low dark channel (<= ~0.25 here); anything above is fog.
    # The batch-level statistics above only match the AVERAGE to the clear references, which left dense-fog photos milky.
    dcp = zero
    if unp.any():
        dcp = F.relu(_dark(pred[unp]).mean((1, 2, 3)) - 0.25).mean()
    # structure
    gpx, gpy = _grad(yp)
    ghx, ghy = _grad(yh)
    mag = (ghx ** 2 + ghy ** 2 + 1e-8).sqrt() + 1e-4
    cos = (gpx * ghx + gpy * ghy) / ((((gpx ** 2 + gpy ** 2 + 1e-8).sqrt() + 1e-4)) * mag)
    struct = ((1 - cos) * (mag / mag.mean((1, 2, 3), keepdim=True))).mean() * 0.5
    # colour: hue direction and saturation
    hue = ((1 - (ao * ai + bo * bi) / (co * ci + 1e-4)) * ci.detach().clamp(max=0.2)).mean()
    sat_loss = F.relu(F.avg_pool2d(sat(hazy), 16) - F.avg_pool2d(sat(pred), 16)).mean()
    total = w["low"] * low + w["tone"] * tone + w["dark"] * dark + w["edge"] * edge + w["struct"] * struct + w["hue"] * hue + w["sat"] * sat_loss + w["noise"] * noise + w["affine"] * affine + w["chroma_cap"] * chroma_cap + w["chroma_ref"] * chroma_ref + w["sat_cap"] * sat_cap + w["dcp"] * dcp
    parts = {"low": low.item(), "tone": tone.item(), "dark": dark.item(), "edge": edge.item(), "struct": struct.item(), "hue": hue.item(), "sat": sat_loss.item(), "noise": noise.item(), "affine": affine.item(), "ccap": chroma_cap.item(), "cref": chroma_ref.item(), "scap": sat_cap.item(), "dcp": dcp.item()}
    return total, parts
