"""
Physically-consistent haze-density augmentation.

Real haze follows I = J*t + A*(1-t) (the atmospheric scattering model),
so any synthetic hazy variant I on disk is anchored to some effective
transmission t along the straight line between the clear image J and I
in pixel space. Alpha-blending clear and hazy tensors,
    synthetic = J + alpha*(I-J),
is algebraically exactly equivalent to sampling a new transmission
t' = 1 - alpha*(1-t) -- not a heuristic, an exact re-parameterization of
the same physical model (expand both forms and they're identical term
for term). This turns each dataset's fixed discrete densities (3 betas
for Cityscapes, 8 variants per plate scene) into a continuous range,
including alpha>1 (denser haze than any variant on disk) and alpha=0 (no
haze at all, teaching the model not to over-correct near-clean input) --
both of which the fixed variants alone can't cover, and both implicated
in the sim-to-real brightness-collapse failure on real photos (e.g.
nn.jpg, where the model's 8 fixed training densities apparently don't
bracket the real haze it was shown).
"""
import random


def augment_haze_density(clear, hazy, alpha_range=(0.0, 1.3)):
    """clear, hazy: tensors in [0,1], same shape. Returns a synthetic hazy
    tensor at a randomly sampled density; clear is always the correct
    target regardless of alpha, so it's returned unchanged."""
    alpha = random.uniform(*alpha_range)
    synthetic = clear + alpha * (hazy - clear)
    return synthetic.clamp(0.0, 1.0)
