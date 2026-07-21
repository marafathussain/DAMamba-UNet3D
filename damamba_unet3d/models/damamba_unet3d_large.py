"""
DAMamba-UNet3D-Large — wide variant for large-scale 3D segmentation.

Uses the same DASSM-CUDA blocks and learned DAS scan as the compact model,
with wider stage widths [48, 96, 192, 384], a 768-d bottleneck, and stacked
encoder DAS blocks (~70M parameters).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .damamba_unet3d import (
    DAMambaBlock3D,
    ConvBlock3d,
    Downsample3d,
    UpBlock3d,
)


class _EncoderStage(nn.Module):
    """Conv block followed by ``depth`` stacked DASSM blocks."""

    def __init__(
        self,
        channels: int,
        depth: int,
        norm: str,
        damamba_cfg: dict,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        blocks: list[nn.Module] = [ConvBlock3d(channels, channels, norm=norm)]
        blocks.extend(DAMambaBlock3D(channels, **damamba_cfg) for _ in range(depth))
        self.blocks = nn.ModuleList(blocks)

    def _run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._run_blocks, x, use_reentrant=False)
        return self._run_blocks(x)


class _DamambaStack(nn.Module):
    """Sequential DASSM blocks with optional per-block checkpointing."""

    def __init__(
        self,
        channels: int,
        depth: int,
        damamba_cfg: dict,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList(
            [DAMambaBlock3D(channels, **damamba_cfg) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class _ConvStack(nn.Module):
    """Stacked ConvBlock3d modules for a convolution-only bottleneck."""

    def __init__(
        self,
        channels: int,
        depth: int,
        norm: str,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList(
            [ConvBlock3d(channels, channels, norm=norm) for _ in range(depth)]
        )

    def _run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._run_blocks, x, use_reentrant=False)
        return self._run_blocks(x)


class DAMambaUNet3DLarge(nn.Module):
    """Wide/deep DAMamba U-Net for large-scale 3D segmentation experiments."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stage_widths: Sequence[int] = (48, 96, 192, 384),
        bottleneck_width: int = 768,
        stage_depths: Sequence[int] = (2, 2, 2, 2),
        bottleneck_depth: int = 4,
        bottleneck_conv_depth: int = 0,
        decoder_damamba_depths: Sequence[int] = (0, 0, 0),
        norm: str = "group",
        damamba_cfg: Optional[dict] = None,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if len(stage_widths) != 4 or len(stage_depths) != 4:
            raise ValueError("stage_widths and stage_depths must have length 4")
        if len(decoder_damamba_depths) != 3:
            raise ValueError("decoder_damamba_depths must have length 3 (dec4/3/2)")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stage_widths = tuple(int(w) for w in stage_widths)
        self.bottleneck_width = int(bottleneck_width)
        self.gradient_checkpointing = gradient_checkpointing

        damamba_cfg = dict(damamba_cfg or {})
        w0, w1, w2, w3 = self.stage_widths

        self.enc1 = ConvBlock3d(in_channels, w0, norm=norm)
        self.down1 = Downsample3d(w0, w1)
        self.enc2 = _EncoderStage(
            w1, int(stage_depths[0]), norm, damamba_cfg, gradient_checkpointing
        )
        self.down2 = Downsample3d(w1, w2)
        self.enc3 = _EncoderStage(
            w2, int(stage_depths[1]), norm, damamba_cfg, gradient_checkpointing
        )
        self.down3 = Downsample3d(w2, w3)
        self.enc4 = _EncoderStage(
            w3, int(stage_depths[2]), norm, damamba_cfg, gradient_checkpointing
        )
        self.down4 = Downsample3d(w3, self.bottleneck_width)
        bn_damamba = int(bottleneck_depth)
        bn_conv = int(bottleneck_conv_depth)
        if bn_damamba > 0 and bn_conv > 0:
            raise ValueError("Set either bottleneck_depth or bottleneck_conv_depth, not both.")
        if bn_damamba > 0:
            self.bottleneck = _DamambaStack(
                self.bottleneck_width,
                bn_damamba,
                damamba_cfg,
                gradient_checkpointing,
            )
        elif bn_conv > 0:
            self.bottleneck = _ConvStack(
                self.bottleneck_width,
                bn_conv,
                norm,
                gradient_checkpointing,
            )
        else:
            self.bottleneck = nn.Identity()

        self.up4 = UpBlock3d(self.bottleneck_width, w3, w3, norm=norm)
        self.dec4_damamba = self._make_damamba_stack(
            w3, int(decoder_damamba_depths[0]), damamba_cfg, gradient_checkpointing
        )
        self.up3 = UpBlock3d(w3, w2, w2, norm=norm)
        self.dec3_damamba = self._make_damamba_stack(
            w2, int(decoder_damamba_depths[1]), damamba_cfg, gradient_checkpointing
        )
        self.up2 = UpBlock3d(w2, w1, w1, norm=norm)
        self.dec2_damamba = self._make_damamba_stack(
            w1, int(decoder_damamba_depths[2]), damamba_cfg, gradient_checkpointing
        )
        self.up1 = UpBlock3d(w1, w0, w0, norm=norm)
        self.out_head = nn.Conv3d(w0, out_channels, kernel_size=1)

    @staticmethod
    def _make_damamba_stack(
        channels: int,
        depth: int,
        damamba_cfg: dict,
        gradient_checkpointing: bool = False,
    ) -> nn.Module:
        if depth <= 0:
            return nn.Identity()
        return _DamambaStack(channels, depth, damamba_cfg, gradient_checkpointing)

    def count_damamba_blocks(self) -> int:
        return sum(1 for m in self.modules() if isinstance(m, DAMambaBlock3D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        x = self.down1(s1)

        s2 = self.enc2(x)
        x = self.down2(s2)

        s3 = self.enc3(x)
        x = self.down3(s3)

        s4 = self.enc4(x)
        x = self.down4(s4)

        x = self.bottleneck(x)

        x = self.up4(x, s4)
        x = self.dec4_damamba(x)
        x = self.up3(x, s3)
        x = self.dec3_damamba(x)
        x = self.up2(x, s2)
        x = self.dec2_damamba(x)
        x = self.up1(x, s1)
        return self.out_head(x)
