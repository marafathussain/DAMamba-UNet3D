"""
Tri-plane Dynamic Adaptive Scan (3D DASSM front-end).

DCNv3 is 2D-only upstream. For volumetric DASSM we apply the official CUDA DAS
on axial, coronal, and sagittal planes and fuse the responses. selective_scan
then runs on the flattened D×H×W token grid.
"""

from __future__ import annotations

import torch
from torch import nn

from .damamba_adaptive_scan import DynamicAdaptiveScan


class TriPlaneDynamicAdaptiveScan(nn.Module):
    """Apply shared 2D DAS (CUDA DCNv3) along three orthogonal orientations."""

    def __init__(self, channels: int, group: int, **das_kwargs):
        super().__init__()
        self.das = DynamicAdaptiveScan(channels=channels, group=group, **das_kwargs)

    def _scan_plane(
        self,
        volume: torch.Tensor,
        plane: str,
    ) -> torch.Tensor:
        """
        volume: (B, C, D, H, W)
        returns: (B, C, D, H, W)
        """
        b, c, d, h, w = volume.shape
        if plane == "axial":
            inp = volume.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w).contiguous()
            x_cl = volume.permute(0, 2, 3, 4, 1).reshape(b * d, h, w, c).contiguous()
            out = self.das(inp, x_cl).view(b, d, c, h, w).permute(0, 2, 1, 3, 4)
        elif plane == "coronal":
            inp = volume.permute(0, 3, 1, 2, 4).reshape(b * h, c, d, w).contiguous()
            x_cl = volume.permute(0, 3, 2, 4, 1).reshape(b * h, d, w, c).contiguous()
            out = self.das(inp, x_cl).view(b, h, c, d, w).permute(0, 2, 3, 1, 4)
        elif plane == "sagittal":
            inp = volume.permute(0, 4, 1, 2, 3).reshape(b * w, c, d, h).contiguous()
            x_cl = volume.permute(0, 4, 2, 3, 1).reshape(b * w, d, h, c).contiguous()
            out = self.das(inp, x_cl).view(b, w, c, d, h).permute(0, 2, 3, 4, 1)
        else:
            raise ValueError(f"Unknown plane '{plane}'")
        return out.contiguous()

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        axial = self._scan_plane(volume, "axial")
        coronal = self._scan_plane(volume, "coronal")
        sagittal = self._scan_plane(volume, "sagittal")
        return (axial + coronal + sagittal) / 3.0
