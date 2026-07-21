#!/usr/bin/env python3
"""Verify CUDA extensions and run a forward/backward pass on GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from damamba_unet3d.models import DAMambaUNet3D, DAMambaUNet3DLarge  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("[FAIL] CUDA is required.", flush=True)
        return 2

    device = torch.device("cuda")

    try:
        import DCNv3  # noqa: F401
        import selective_scan_cuda_oflex_rh  # noqa: F401
    except ImportError:
        print("[FAIL] Missing CUDA extensions. Run: python scripts/install_ops.py", flush=True)
        return 1

    damamba_cfg = {"d_state": 16, "d_conv": 4, "expand": 2, "scan_mode": "das"}

    compact = DAMambaUNet3D(
        in_channels=4,
        out_channels=4,
        base_width=16,
        use_damamba_in=["E2", "E3", "E4"],
        bottleneck_depth=1,
        damamba_cfg=damamba_cfg,
    ).to(device)
    n_compact = sum(p.numel() for p in compact.parameters())
    x = torch.randn(1, 4, 64, 128, 128, device=device)
    y = compact(x)
    loss = y.mean()
    loss.backward()
    print(
        f"[ OK ] DAMambaUNet3D  params={n_compact/1e6:.2f}M  "
        f"blocks={compact.count_damamba_blocks()}  out={tuple(y.shape)}",
        flush=True,
    )

    large = DAMambaUNet3DLarge(
        in_channels=4,
        out_channels=4,
        stage_widths=[48, 96, 192, 384],
        bottleneck_width=768,
        stage_depths=[2, 2, 2, 2],
        bottleneck_depth=0,
        bottleneck_conv_depth=1,
        damamba_cfg={**damamba_cfg, "expand": 4, "head_dim": 16},
    ).to(device)
    n_large = sum(p.numel() for p in large.parameters())
    x_large = torch.randn(1, 4, 64, 64, 64, device=device)
    y_large = large(x_large)
    print(
        f"[ OK ] DAMambaUNet3DLarge  params={n_large/1e6:.2f}M  "
        f"blocks={large.count_damamba_blocks()}  out={tuple(y_large.shape)}",
        flush=True,
    )

    print("[ OK ] All checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
