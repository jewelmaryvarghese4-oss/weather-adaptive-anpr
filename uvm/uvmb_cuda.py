"""
UVMB using the REAL `mamba_ssm.Mamba` (fused CUDA/Triton kernel), for
training on a CUDA-capable box -- e.g. the WSL2 setup you already have
`mamba_ssm` installed in.

This is otherwise identical to `uvmb.py` (the pure-PyTorch version): same
two documented deviations from the original repo (proper flatten+transpose
instead of their bare `.reshape()`, explicit `softmax(dim=-1)`), same
constructor shape. The only difference is the import at the top.

Requirements (matches the original repo's README):
    pip install causal-conv1d>=1.1.0
    pip install mamba-ssm
    Linux (WSL2 counts), NVIDIA GPU, PyTorch 1.12+, CUDA 11.6+

Use this for training. Use `uvmb.py` (pure-PyTorch) for anything that needs
to run without a CUDA-kernel dependency -- CPU dev boxes, and eventually
ONNX export for the Jetson deployment path, since `mamba_ssm`'s kernel has
no ONNX export path (the reason `mamba_bissm.py` exists in the first
place). Train here, export from the pure-PyTorch version once weights are
ported over -- the two `SSMLayer`/`Mamba` implementations have the same
parameter shapes (same d_model/d_state/d_conv/expand), so a state_dict
saved from one should load into the other; NOT yet verified, check this
before relying on it.
"""
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as e:
    raise ImportError(
        "mamba_ssm is not importable in this environment. Install with "
        "`pip install causal-conv1d>=1.1.0` then `pip install mamba-ssm` "
        "on a CUDA-capable Linux/WSL2 box -- or use uvmb.py (the "
        "pure-PyTorch version, no CUDA kernel needed) instead."
    ) from e


class UVMBCuda(nn.Module):
    def __init__(self, c=3, w=64, h=64, use_original_reshape_bug=False):
        super().__init__()
        self.use_original_reshape_bug = use_original_reshape_bug

        self.convb = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=c, kernel_size=3, stride=1, padding=1),
        )
        self.model1 = Mamba(d_model=c, d_state=16, d_conv=4, expand=2)
        self.model2 = Mamba(d_model=c, d_state=16, d_conv=4, expand=2)
        self.model3 = Mamba(d_model=w * h, d_state=16, d_conv=4, expand=2)

        self.smooth = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.ln = nn.LayerNorm(normalized_shape=c)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, w, h = x.shape
        x = self.convb(x) + x

        if self.use_original_reshape_bug:
            x = self.ln(x.reshape(b, -1, c))
        else:
            x = self.ln(x.flatten(2).transpose(1, 2))

        y = self.model1(x).permute(0, 2, 1)
        z = self.model3(y).permute(0, 2, 1)
        att = self.softmax(self.model2(x))
        result = att * z

        if self.use_original_reshape_bug:
            output = result.reshape(b, c, w, h)
        else:
            output = result.transpose(1, 2).reshape(b, c, w, h)

        return self.smooth(output)
