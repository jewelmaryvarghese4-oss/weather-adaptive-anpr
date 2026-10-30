"""
mamba_blocks.py — patched to use the official mamba_ssm fused CUDA
kernel for the scan when it's installed and you're on GPU, falling
back to your existing pure-PyTorch loop otherwise.

Nothing else changes: SS2D, VSSBlock, and every other file that imports
from here are untouched. This is a surgical swap inside
SelectiveScan1D.forward only — same class name, same __init__ params,
same input/output shapes (B, L, D) -> (B, L, D). model.py, losses.py,
train.py etc. need zero changes.

Install (on your laptop, with your CUDA GPU):
    pip install mamba-ssm causal-conv1d

Heads-up: mamba-ssm ships prebuilt wheels for common CUDA/torch/Python
combos on Linux; on native Windows there's often no prebuilt wheel and
pip will try to build the CUDA extension from source, which needs a
matching CUDA toolkit (nvcc) + MSVC build tools on PATH and can fail in
annoying ways. If `pip install mamba-ssm causal-conv1d` errors out
trying to compile, the much more reliable path on Windows is installing
inside WSL2 (Windows Subsystem for Linux) instead of native Windows —
WSL2 presents as Linux to pip, so the prebuilt wheels apply normally.

Either way, this file works fine without mamba_ssm installed at all —
it just silently uses the slower pure-PyTorch path (exactly what you've
already trained and validated with). So it's safe to drop in right now
regardless of whether the pip install succeeds.

IMPORTANT: before trusting a real training run on the fused path, run
verify_fused_scan_parity.py once — it checks the fused kernel produces
the same output as your existing loop (up to floating-point tolerance)
on a real forward pass, so a shape/argument mistake in this integration
gets caught immediately instead of silently corrupting training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _HAS_FUSED_SCAN = True
except ImportError:
    _HAS_FUSED_SCAN = False


class SelectiveScan1D(nn.Module):
    def __init__(self, d_model, d_state=16, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_model // 16)

        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

    def _scan_loop(self, x, delta, A, Bmat, Cmat):
        """Your original pure-PyTorch sequential scan — used as the
        fallback when the fused CUDA kernel isn't available/applicable,
        and as the ground truth that verify_fused_scan_parity.py checks
        the fused path against."""
        B, L, D = x.shape
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB_x = delta.unsqueeze(-1) * Bmat.unsqueeze(2) * x.unsqueeze(-1)

        h = x.new_zeros(B, D, self.d_state)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_x[:, t]
            y_t = torch.einsum("bdn,bn->bd", h, Cmat[:, t])
            ys.append(y_t)
        return torch.stack(ys, dim=1)

    def _scan_fused(self, x, delta, A, Bmat, Cmat):
        """Official fused CUDA kernel. Note the different memory layout
        it expects — (B, D, L), channels-before-length — vs. the (B, L, D)
        layout used everywhere else in this codebase, hence the
        transposes in and out."""
        u = x.transpose(1, 2).contiguous()            # (B, D, L)
        delta_t = delta.transpose(1, 2).contiguous()    # (B, D, L)
        B_t = Bmat.transpose(1, 2).contiguous()          # (B, N, L)
        C_t = Cmat.transpose(1, 2).contiguous()          # (B, N, L)

        y = selective_scan_fn(
            u, delta_t, A, B_t, C_t,
            D=None,              # keep the D-skip term applied manually below,
            z=None,              # matching your existing code exactly
            delta_bias=None,
            delta_softplus=False,  # already applied to `delta` before this call
        )  # (B, D, L)
        return y.transpose(1, 2)  # back to (B, L, D)

    def forward(self, x):
        B, L, D = x.shape
        N = self.d_state

        x_dbl = self.x_proj(x)
        delta, Bmat, Cmat = torch.split(x_dbl, [self.dt_rank, N, N], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        A = -torch.exp(self.A_log)

        if _HAS_FUSED_SCAN and x.is_cuda:
            y = self._scan_fused(x, delta, A, Bmat, Cmat)
        else:
            y = self._scan_loop(x, delta, A, Bmat, Cmat)

        return y + x * self.D


class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.act = nn.SiLU()
        self.conv2d = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.scan = SelectiveScan1D(d_model, d_state=d_state)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _four_directions(self, x):
        B, H, W, C = x.shape
        row_major = x.reshape(B, H * W, C)
        row_major_rev = torch.flip(row_major, dims=[1])
        col_major = x.permute(0, 2, 1, 3).reshape(B, H * W, C)
        col_major_rev = torch.flip(col_major, dims=[1])
        return row_major, row_major_rev, col_major, col_major_rev, H, W

    def forward(self, x):
        B, H, W, C = x.shape

        x_and_gate = self.in_proj(x)
        x_ssm, gate = x_and_gate.chunk(2, dim=-1)

        x_conv = self.conv2d(x_ssm.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_conv = self.act(x_conv)

        seqs = self._four_directions(x_conv)
        rm, rm_rev, cm, cm_rev, H, W = seqs

        y_rm = self.scan(rm)
        y_rm_rev = torch.flip(self.scan(rm_rev), dims=[1])
        y_cm = self.scan(cm)
        y_cm_rev = torch.flip(self.scan(cm_rev), dims=[1])

        y_cm = y_cm.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, H * W, C)
        y_cm_rev = y_cm_rev.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, H * W, C)

        y = (y_rm + y_rm_rev + y_cm + y_cm_rev) / 4.0
        y = y.reshape(B, H, W, C)

        y = self.out_norm(y)
        y = y * self.act(gate)
        y = self.out_proj(y)
        return y


class VSSBlock(nn.Module):
    def __init__(self, d_model, d_state=16, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ss2d = SS2D(d_model, d_state=d_state)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        x = x + self.ss2d(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
