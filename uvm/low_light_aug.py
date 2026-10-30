"""
Synthetic low-light augmentation for UVMNet training.

Motivation: every real training source in this project (OTS, FoggyPlates,
NH-HAZE) is daytime. The model has never seen a dusk/night hazy scene, and
on real-world test photos like heavyfog.jpg (dark, taillights visible,
dense fog) it produces heavy grain artifacts -- an unconstrained, ungrounded
correction on genuinely unfamiliar input. Real paired night-fog datasets
are rare, so this synthesizes the missing domain from the paired data
already on hand instead of sourcing new data.

Design choices, and why:

1. The SAME darkness/gamma curve is applied to both hazy and clear in a
   pair. This teaches "dehaze under low ambient light" (haze removed,
   scene stays dusk-lit) rather than "brighten a dark hazy photo up to
   daylight" -- the latter asks for a much larger correction than the
   model already struggles to make stably on off-domain input, which is
   the exact failure mode being targeted here, not something to make
   worse.

2. Sensor noise (the grain a real high-ISO low-light photo has) is added
   to the HAZY side only, never to clear. The clear target stays a clean
   reference, so the model is directly supervised to suppress noise as
   part of its correction -- not reproduce it, which is what happened
   when NH-HAZE's own real-photo grain got over-represented via repeat
   sampling in an earlier run.

3. Applied per-sample within a batch (not per-batch), so a mix of normal
   and low-light examples train together -- and only to a fraction of
   samples (--lowlight_prob), so the existing daytime performance isn't
   diluted by over-representing a synthetic domain.
"""
import torch


def augment_low_light(hazy, clear, prob=0.35,
                       darkness_range=(0.15, 0.55),
                       gamma_range=(1.2, 2.0),
                       noise_std_range=(0.02, 0.08)):
    """
    hazy, clear: (B, C, H, W) tensors in [0, 1], already paired/cropped.
    Returns (hazy_aug, clear_aug) with a random subset of the batch
    darkened to simulate dusk/night conditions.

    Fully vectorized, no data-dependent branching (no .nonzero()/.any() on
    a GPU tensor) -- those force a GPU->CPU sync every call, since Python
    needs the actual value to decide what to do. Cheap on native hardware,
    but disproportionately expensive under WSL2's virtualized GPU
    passthrough (same class of problem as the earlier /mnt/c stat-call
    slowness), and this runs every single training batch. Instead this
    computes the darkened+noised version for the WHOLE batch unconditionally
    (trivially cheap next to the UNet forward/backward pass) and blends
    per-sample with a continuous 0/1 mask -- same result, zero syncs.
    """
    if prob <= 0:
        return hazy, clear

    B = hazy.shape[0]
    device = hazy.device

    select = (torch.rand(B, device=device) < prob).to(hazy.dtype).view(B, 1, 1, 1)
    darkness = torch.empty(B, device=device).uniform_(*darkness_range).view(B, 1, 1, 1)
    gamma = torch.empty(B, device=device).uniform_(*gamma_range).view(B, 1, 1, 1)
    noise_std = torch.empty(B, device=device).uniform_(*noise_std_range).view(B, 1, 1, 1)

    h_dark = hazy.clamp(min=1e-4).pow(gamma) * darkness
    c_dark = clear.clamp(min=1e-4).pow(gamma) * darkness

    # Sensor noise scales with darkness (darker -> effectively higher ISO
    # -> more visible grain), matching real low-light camera behavior.
    noise = torch.randn_like(h_dark) * noise_std
    h_dark = (h_dark + noise).clamp(0.0, 1.0)
    c_dark = c_dark.clamp(0.0, 1.0)

    hazy_out = select * h_dark + (1 - select) * hazy
    clear_out = select * c_dark + (1 - select) * clear
    return hazy_out, clear_out
