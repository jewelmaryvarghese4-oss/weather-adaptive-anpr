"""
Differentiable guided filter for the output of the dehazer (guide = the hazy input's luminance, stretched to 0..1 per image).
Same recipe as the post-processing that gave the best plate-reading gains (radius 4, eps 1e-2); here it is a torch layer so the dehazer can be
trained THROUGH it. Computed in float32 (box filters via average pooling with reflect padding).

    out = guided_filter_head(model_output, hazy_input, radius=4, eps=1e-2)      # both (B, 3, H, W), values around 0..1
"""
import torch
import torch.nn.functional as F


def _box(x, r):
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="reflect"), 2 * r + 1, stride=1)


def guided_filter_head(pred, hazy, radius=4, eps=1e-2):
    with torch.autocast(device_type="cuda", enabled=False):
        p = pred.float()
        g = (0.299 * hazy[:, 0:1] + 0.587 * hazy[:, 1:2] + 0.114 * hazy[:, 2:3]).float()
        lo = g.amin(dim=(2, 3), keepdim=True)
        hi = g.amax(dim=(2, 3), keepdim=True)
        g = (g - lo) / (hi - lo).clamp(min=1e-3)
        mi = _box(g, radius)
        var = _box(g * g, radius) - mi * mi
        out = []
        for c in range(p.shape[1]):
            pc = p[:, c:c + 1]
            mp = _box(pc, radius)
            a = (_box(g * pc, radius) - mi * mp) / (var + eps)
            b = mp - a * mi
            out.append(_box(a, radius) * g + _box(b, radius))
        return torch.cat(out, dim=1)
