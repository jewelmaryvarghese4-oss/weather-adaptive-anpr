"""
verify_fused_scan_parity.py

Run this ONCE on your laptop (GPU, after `pip install mamba-ssm
causal-conv1d`) before trusting a real training run on the fused-kernel
path. It checks that mamba_blocks.py's fused CUDA path produces the
same result as the original pure-PyTorch loop — both forward output AND
gradients — on identical inputs/weights. This exists because a subtle
shape/argument mistake in wiring up selective_scan_fn (wrong tensor
layout, double-applying softplus, double-adding the D-skip term, etc.)
would silently corrupt training rather than crash it — much better to
catch that here on one forward+backward pass than after a multi-hour run.

Usage:
    python verify_fused_scan_parity.py
"""

import torch

from mamba_blocks import SelectiveScan1D, _HAS_FUSED_SCAN


def main():
    if not _HAS_FUSED_SCAN:
        print("mamba_ssm is not importable in this environment — nothing "
              "to verify (the model will just use the pure-PyTorch loop, "
              "which is already validated). Install mamba-ssm and re-run "
              "this script to actually test the fused path.")
        return

    if not torch.cuda.is_available():
        print("CUDA not available — the fused kernel only activates on "
              "GPU tensors, so there's nothing to compare here either.")
        return

    device = torch.device("cuda")
    torch.manual_seed(0)

    d_model, d_state, seq_len, batch = 32, 16, 64, 2
    scan = SelectiveScan1D(d_model, d_state=d_state).to(device)

    x = torch.randn(batch, seq_len, d_model, device=device, requires_grad=True)
    x_clone = x.detach().clone().requires_grad_(True)

    # Run the exact same shared projections (x_proj/dt_proj/A_log) through
    # both paths by calling the same module's internal pieces directly,
    # so any difference is purely from _scan_loop vs _scan_fused.
    def run(scan_module, x_in, use_fused: bool):
        x_dbl = scan_module.x_proj(x_in)
        delta, Bmat, Cmat = torch.split(
            x_dbl, [scan_module.dt_rank, scan_module.d_state, scan_module.d_state], dim=-1
        )
        delta = torch.nn.functional.softplus(scan_module.dt_proj(delta))
        A = -torch.exp(scan_module.A_log)
        if use_fused:
            y = scan_module._scan_fused(x_in, delta, A, Bmat, Cmat)
        else:
            y = scan_module._scan_loop(x_in, delta, A, Bmat, Cmat)
        return y + x_in * scan_module.D

    y_loop = run(scan, x, use_fused=False)
    y_fused = run(scan, x_clone, use_fused=True)

    fwd_diff = (y_loop - y_fused).abs().max().item()
    print(f"forward max abs diff (loop vs fused): {fwd_diff:.6e}")

    y_loop.sum().backward()
    y_fused.sum().backward()

    grad_diff = (x.grad - x_clone.grad).abs().max().item()
    print(f"input-gradient max abs diff (loop vs fused): {grad_diff:.6e}")

    ok = fwd_diff < 1e-2 and grad_diff < 1e-2
    print()
    if ok:
        print("PASS — fused kernel matches the pure-PyTorch loop closely "
              "enough to trust for training. (Some floating-point "
              "difference is normal/expected between a fused CUDA kernel "
              "and a naive loop — this just confirms it's not wired up "
              "wrong.)")
    else:
        print("FAIL — fused and loop paths disagree by more than expected. "
              "Do NOT train on the fused path yet — something in the "
              "shape/argument wiring in mamba_blocks.py's _scan_fused is "
              "likely wrong. Send me this output and I'll help debug it.")


if __name__ == "__main__":
    main()
