"""
DAMamba-UNet (3D) — tri-plane DASSM-CUDA + volumetric selective_scan.
"""

from __future__ import annotations

from typing import Optional, Sequence, Set

import torch
from torch import nn

_BACKEND_BANNER_PRINTED_3D = False


def _print_backend_banner_3d() -> None:
    global _BACKEND_BANNER_PRINTED_3D
    if _BACKEND_BANNER_PRINTED_3D:
        return
    _BACKEND_BANNER_PRINTED_3D = True
    try:
        from .damamba_adaptive_scan import _DCN_VERSION
    except Exception:
        _DCN_VERSION = 0.0
    try:
        import selective_scan_cuda_oflex_rh as _ss
        ss_path = getattr(_ss, "__file__", "<built-in>")
    except Exception:
        try:
            from ..ops.selective_scan import selective_scan_cuda_oflex_rh as _ss
            ss_path = getattr(_ss, "__file__", "<built-in>")
        except Exception:
            ss_path = "<unavailable>"
    print(
        f"[DAMambaBlock3D] backend=DASSM3D-CUDA (tri-plane DAS + 3D SSM) "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"DCNv3={_DCN_VERSION} selective_scan_cuda_oflex_rh={ss_path}",
        flush=True,
    )


class DAMambaBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        head_dim: int = 16,
        **kwargs,
    ):
        super().__init__()
        from .damamba_dassm3d import DASSM3D

        _print_backend_banner_3d()
        self.channels = channels
        self.block = DASSM3D(
            d_model=channels,
            head_dim=head_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            scan_mode=kwargs.get("scan_mode", "das"),
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()
        x_norm = self.norm(x_cl).permute(0, 4, 1, 2, 3).contiguous()
        y = self.block(x_norm)
        return self.dropout(y) + x


def make_norm3d(norm: str, channels: int) -> nn.Module:
    if norm == "group":
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return nn.GroupNorm(groups, channels)
        return nn.GroupNorm(1, channels)
    if norm == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    return nn.BatchNorm3d(channels)


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "group"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            make_norm3d(norm, out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            make_norm3d(norm, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Upsample3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UpBlock3d(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm: str = "group"):
        super().__init__()
        self.up = Upsample3d(in_channels, out_channels)
        self.conv = ConvBlock3d(out_channels + skip_channels, out_channels, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            d = min(x.shape[2], skip.shape[2])
            h = min(x.shape[3], skip.shape[3])
            w = min(x.shape[4], skip.shape[4])
            x = x[:, :, :d, :h, :w]
            skip = skip[:, :, :d, :h, :w]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DAMambaUNet3D(nn.Module):
    """3D U-Net with tri-plane DASSM-CUDA blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_width: int = 16,
        use_damamba_in: Optional[Sequence[str]] = None,
        decoder_damamba: bool = False,
        bottleneck_depth: int = 2,
        norm: str = "group",
        damamba_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_width = base_width
        self.use_damamba_in: Set[str] = set(s.upper() for s in (use_damamba_in or []))
        self.decoder_damamba = decoder_damamba
        self.bottleneck_depth = bottleneck_depth

        c = base_width
        damamba_cfg = damamba_cfg or {}
        self.enc1 = ConvBlock3d(in_channels, c, norm=norm)
        self.down1 = Downsample3d(c, 2 * c)

        self.enc2 = ConvBlock3d(2 * c, 2 * c, norm=norm)
        self.damamba2 = (
            DAMambaBlock3D(2 * c, **damamba_cfg) if "E2" in self.use_damamba_in else nn.Identity()
        )
        self.down2 = Downsample3d(2 * c, 4 * c)

        self.enc3 = ConvBlock3d(4 * c, 4 * c, norm=norm)
        self.damamba3 = (
            DAMambaBlock3D(4 * c, **damamba_cfg) if "E3" in self.use_damamba_in else nn.Identity()
        )
        self.down3 = Downsample3d(4 * c, 8 * c)

        self.enc4 = ConvBlock3d(8 * c, 8 * c, norm=norm)
        self.damamba4 = (
            DAMambaBlock3D(8 * c, **damamba_cfg) if "E4" in self.use_damamba_in else nn.Identity()
        )
        self.down4 = Downsample3d(8 * c, 16 * c)

        bottleneck_blocks = []
        if "BOTTLENECK" in self.use_damamba_in:
            for _ in range(bottleneck_depth):
                bottleneck_blocks.append(DAMambaBlock3D(16 * c, **damamba_cfg))
        self.bottleneck = nn.Sequential(*bottleneck_blocks) if bottleneck_blocks else nn.Identity()

        self.up4 = UpBlock3d(16 * c, 8 * c, 8 * c, norm=norm)
        self.dec4_damamba = (
            DAMambaBlock3D(8 * c, **damamba_cfg) if decoder_damamba else nn.Identity()
        )
        self.up3 = UpBlock3d(8 * c, 4 * c, 4 * c, norm=norm)
        self.up2 = UpBlock3d(4 * c, 2 * c, 2 * c, norm=norm)
        self.up1 = UpBlock3d(2 * c, c, c, norm=norm)

        self.out_head = nn.Conv3d(c, out_channels, kernel_size=1)

    def count_damamba_blocks(self) -> int:
        return sum(1 for m in self.modules() if isinstance(m, DAMambaBlock3D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        x = self.down1(s1)

        s2 = self.enc2(x)
        s2 = self.damamba2(s2)
        x = self.down2(s2)

        s3 = self.enc3(x)
        s3 = self.damamba3(s3)
        x = self.down3(s3)

        s4 = self.enc4(x)
        s4 = self.damamba4(s4)
        x = self.down4(s4)

        x = self.bottleneck(x)

        x = self.up4(x, s4)
        x = self.dec4_damamba(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.out_head(x)
