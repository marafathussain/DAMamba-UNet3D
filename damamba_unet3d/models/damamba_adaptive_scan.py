"""
Dynamic Adaptive Scan (DAS) from DAMamba.

Hard requirement: the DCNv3 CUDA extension MUST be importable. There is no
PyTorch fallback. If you see an ImportError here, you have not built the
upstream DAMamba CUDA kernels. See docs/USE_DCN_CUDA.md or run:

    python scripts/install_damamba_ops.py
    python scripts/verify_dassm.py
"""

from typing import Optional

import torch
from torch import nn
from torch.nn.init import constant_
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import DCNv3
except ImportError as exc:
    raise ImportError(
        "DCNv3 CUDA extension is required for damamba_unet3d.models.damamba_adaptive_scan "
        "and DAMamba-UNet, but it is not importable. PyTorch fallback is disabled "
        "by design (see reviews.txt R2). Build the kernel on a GPU node with:\n"
        "    python scripts/install_ops.py\n"
        "Then verify with:\n"
        "    python scripts/verify_dassm.py"
    ) from exc

def _get_dcn_version() -> float:
    """DCNv3 1.1 requires remove_center in dcnv3_forward; detect installed version."""
    try:
        from importlib.metadata import version as _pkg_version
        return float(_pkg_version("DCNv3"))
    except Exception:
        pass
    try:
        import pkg_resources
        return float(pkg_resources.get_distribution("DCNv3").version)
    except Exception:
        pass
    return 1.1  # install_damamba_ops.py builds DCNv3 1.1 from the pinned upstream commit


_DCN_VERSION = _get_dcn_version()


class DCNv3Function(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dilation_h: int,
        dilation_w: int,
        group: int,
        group_channels: int,
        offset_scale: float,
        im2col_step: int,
        remove_center: int,
    ) -> torch.Tensor:
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center

        args = [
            input,
            offset,
            mask,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            group_channels,
            offset_scale,
            ctx.im2col_step,
        ]
        if remove_center or _DCN_VERSION > 1.0:
            args.append(remove_center)

        output = DCNv3.dcnv3_forward(*args)
        ctx.save_for_backward(input, offset, mask)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        input, offset, mask = ctx.saved_tensors

        args = [
            input,
            offset,
            mask,
            ctx.kernel_h,
            ctx.kernel_w,
            ctx.stride_h,
            ctx.stride_w,
            ctx.pad_h,
            ctx.pad_w,
            ctx.dilation_h,
            ctx.dilation_w,
            ctx.group,
            ctx.group_channels,
            ctx.offset_scale,
            grad_output.contiguous(),
            ctx.im2col_step,
        ]
        if ctx.remove_center or _DCN_VERSION > 1.0:
            args.append(ctx.remove_center)

        grad_input, grad_offset, grad_mask = DCNv3.dcnv3_backward(*args)
        return (
            grad_input,
            grad_offset,
            grad_mask,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class to_channels_first(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)


def build_norm_layer(
    dim: int,
    norm_layer: str,
    in_format: str = "channels_last",
    out_format: str = "channels_last",
    eps: float = 1e-6,
) -> nn.Module:
    layers = []
    if norm_layer == "BN":
        if in_format == "channels_last":
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == "channels_last":
            layers.append(to_channels_last())
    elif norm_layer == "LN":
        if in_format == "channels_first":
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == "channels_first":
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f"Unsupported norm_layer {norm_layer}")
    return nn.Sequential(*layers)


def build_act_layer(act_layer: str) -> nn.Module:
    if act_layer == "ReLU":
        return nn.ReLU(inplace=True)
    if act_layer == "SiLU":
        return nn.SiLU(inplace=True)
    if act_layer == "GELU":
        return nn.GELU()
    raise NotImplementedError(f"Unsupported act_layer {act_layer}")


class CenterFeatureScaleModule(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        center_feature_scale_proj_weight: torch.Tensor,
        center_feature_scale_proj_bias: torch.Tensor,
    ) -> torch.Tensor:
        center_feature_scale = torch.nn.functional.linear(
            query,
            weight=center_feature_scale_proj_weight,
            bias=center_feature_scale_proj_bias,
        ).sigmoid()
        return center_feature_scale


class DynamicAdaptiveScan(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 1,
        dw_kernel_size: Optional[int] = 3,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        group: int = 1,
        offset_scale: float = 1.0,
        act_layer: str = "GELU",
        norm_layer: str = "LN",
        center_feature_scale: bool = False,
        remove_center: bool = False,
    ):
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f"channels must be divisible by group, but got {channels} and {group}"
            )
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size

        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError("remove_center is only compatible with odd kernel size.")

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels,
            ),
            build_norm_layer(channels, norm_layer, "channels_first", "channels_last"),
            build_act_layer(act_layer),
        )
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - self.remove_center) * 2,
        )
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float)
            )
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group)
            )
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self) -> None:
        constant_(self.offset.weight.data, 0.0)
        constant_(self.offset.bias.data, 0.0)

    def forward(self, input: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = input.shape
        x_proj = x
        x1 = self.dw_conv(input)
        offset = self.offset(x1)
        mask = torch.ones(n, h, w, self.group, device=x.device, dtype=x.dtype)

        x = DCNv3Function.apply(
            x,
            offset,
            mask,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.pad,
            self.pad,
            self.dilation,
            self.dilation,
            self.group,
            self.group_channels,
            self.offset_scale,
            256,
            self.remove_center,
        )

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias
            )
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group
            ).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        return x.permute(0, 3, 1, 2).contiguous()
