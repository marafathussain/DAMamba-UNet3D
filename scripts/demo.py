#!/usr/bin/env python3
"""Minimal forward-pass demo for DAMamba-UNet3D models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from damamba_unet3d.models import DAMambaUNet3D, DAMambaUNet3DLarge  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DAMamba-UNet3D forward demo")
    p.add_argument("--model", choices=["compact", "large"], default="compact")
    p.add_argument("--in-channels", type=int, default=4)
    p.add_argument("--out-channels", type=int, default=4)
    p.add_argument("--depth", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    shape = (1, args.in_channels, args.depth, args.height, args.width)
    x = torch.randn(*shape, device=device)

    damamba_cfg = {"d_state": 16, "d_conv": 4, "expand": 2, "scan_mode": "das"}

    if args.model == "compact":
        model = DAMambaUNet3D(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            base_width=16,
            use_damamba_in=["E2", "E3", "E4"],
            bottleneck_depth=1,
            damamba_cfg=damamba_cfg,
        )
    else:
        model = DAMambaUNet3DLarge(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            stage_widths=[48, 96, 192, 384],
            bottleneck_width=768,
            stage_depths=[2, 2, 2, 2],
            bottleneck_depth=0,
            bottleneck_conv_depth=1,
            damamba_cfg={**damamba_cfg, "expand": 4, "head_dim": 16},
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        y = model(x)

    print(f"model={args.model}")
    print(f"input={tuple(x.shape)}  output={tuple(y.shape)}")
    print(f"parameters={n_params:,} ({n_params/1e6:.2f}M)")
    print(f"damamba_blocks={model.count_damamba_blocks()}")


if __name__ == "__main__":
    main()
