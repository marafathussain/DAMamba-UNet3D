"""Deterministic 3D token orderings for fixed-scan ablations.

Volumes are (B, C, D, H, W). Default ``selective_scan`` flatten order is
raster: index = d * (H*W) + h * W + w  (w fastest).

These helpers build gather indices for alternative scan patterns and for
tri-orient flattenings (axial / coronal / sagittal major axis).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

PermKey = Tuple[int, int, int, str]


def _linear_index(d: int, h: int, w: int, di: int, hi: int, wi: int) -> int:
    return di * h * w + hi * w + wi


def raster_indices(d: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    return torch.arange(d * h * w, device=device, dtype=torch.long)


def snake_indices(d: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Within each axial slice, alternate row direction (classic snake)."""
    out = []
    for di in range(d):
        for hi in range(h):
            cols = range(w) if hi % 2 == 0 else range(w - 1, -1, -1)
            for wi in cols:
                out.append(_linear_index(d, h, w, di, hi, wi))
    return torch.tensor(out, device=device, dtype=torch.long)


def cross_indices(d: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Center-out radial order within each axial slice (cross / spiral-like)."""
    cx = (h - 1) / 2.0
    cy = (w - 1) / 2.0
    ranked = []
    for di in range(d):
        slice_entries = []
        for hi in range(h):
            for wi in range(w):
                dist = (hi - cx) ** 2 + (wi - cy) ** 2
                slice_entries.append((dist, _linear_index(d, h, w, di, hi, wi)))
        slice_entries.sort(key=lambda t: t[0])
        ranked.extend(idx for _, idx in slice_entries)
    return torch.tensor(ranked, device=device, dtype=torch.long)


def orient_indices(
    d: int, h: int, w: int, orient: str, device: torch.device
) -> torch.Tensor:
    """Major-axis flattening for tri-orient scans."""
    out = []
    if orient == "axial":
        for di in range(d):
            for hi in range(h):
                for wi in range(w):
                    out.append(_linear_index(d, h, w, di, hi, wi))
    elif orient == "coronal":
        for hi in range(h):
            for di in range(d):
                for wi in range(w):
                    out.append(_linear_index(d, h, w, di, hi, wi))
    elif orient == "sagittal":
        for wi in range(w):
            for di in range(d):
                for hi in range(h):
                    out.append(_linear_index(d, h, w, di, hi, wi))
    else:
        raise ValueError(f"Unknown orient '{orient}'")
    return torch.tensor(out, device=device, dtype=torch.long)


def inverse_indices(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv


def get_scan_indices(
    d: int,
    h: int,
    w: int,
    mode: str,
    device: torch.device,
    cache: Dict[PermKey, torch.Tensor],
) -> torch.Tensor:
    key = (d, h, w, mode)
    if key not in cache:
        if mode == "raster":
            cache[key] = raster_indices(d, h, w, device)
        elif mode == "snake":
            cache[key] = snake_indices(d, h, w, device)
        elif mode == "cross":
            cache[key] = cross_indices(d, h, w, device)
        elif mode in ("axial", "coronal", "sagittal"):
            cache[key] = orient_indices(d, h, w, mode, device)
        else:
            raise ValueError(f"Unknown fixed scan mode '{mode}'")
    return cache[key]
