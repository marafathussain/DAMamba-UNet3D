"""
3D DASSM: Conv3d + tri-plane CUDA DAS + selective_scan on D×H×W tokens.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
from torch import nn

from .damamba_adaptive_scan3d import TriPlaneDynamicAdaptiveScan
from .scan_patterns import get_scan_indices, inverse_indices

try:
    from ..ops.selective_scan.utils import selective_scan_fn
except ImportError as exc:
    raise ImportError(
        "selective_scan_cuda_oflex_rh CUDA extension is required for DASSM3D. "
        "Build with: python scripts/install_ops.py"
    ) from exc


def _same_size_conv_pad3d(kernel_size: int) -> tuple[int, int, int, int, int, int]:
    total = kernel_size - 1
    left = total // 2
    right = total - left
    return left, right, left, right, left, right


class LayerNorm3d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class DASSM3D(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int = 16,
        d_state: int = 1,
        d_conv: int = 3,
        expand: int = 1,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        scan_mode: str = "das",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.scan_mode = scan_mode.lower()
        self._perm_cache: dict = {}
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Conv3d(self.d_model, self.d_inner, 1, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=0,
            **factory_kwargs,
        )
        self._conv_pad = _same_size_conv_pad3d(d_conv)
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self._dt_init(
            self.dt_rank,
            self.d_inner,
            dt_scale,
            dt_init,
            dt_min,
            dt_max,
            dt_init_floor,
            **factory_kwargs,
        )
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self._A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self._D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn
        self.out_norm = LayerNorm3d(self.d_inner)
        self.out_proj = nn.Conv3d(self.d_inner, self.d_model, 1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        num_group = max(1, d_model // head_dim)
        if self.scan_mode == "das":
            self.da_scan = TriPlaneDynamicAdaptiveScan(
                channels=self.d_inner, group=num_group
            )
        else:
            self.da_scan = nn.Identity()

    @staticmethod
    def _dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float,
        dt_init: str,
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
        **factory_kwargs,
    ) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        else:
            raise NotImplementedError(dt_init)
        return dt_proj

    @staticmethod
    def _A_log_init(d_state: int, d_inner: int, init: str):
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1).contiguous()
        A_log = nn.Parameter(torch.log(A))
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner: int, init: str = "random"):
        D = nn.Parameter(torch.ones(d_inner))
        D._no_weight_decay = True
        return D

    def _run_selective_scan(self, xs: torch.Tensor) -> torch.Tensor:
        """xs: (B, C, L)"""
        b, c, l = xs.shape
        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, c), xs)
        dts, bs, cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1
        )
        dts = torch.matmul(self.dt_projs_weight.view(1, c, -1), dts)

        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        h_out = self.selective_scan(
            xs,
            dts,
            As,
            bs,
            None,
            z=None,
            delta_bias=self.dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        y = (h_out * cs.unsqueeze(1)).sum(dim=2)
        y = y + xs * Ds.view(-1, 1)
        return y

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        l = d * h * w
        xs = x.view(b, -1, l)

        if self.scan_mode == "tri_orient":
            outs = []
            for orient in ("axial", "coronal", "sagittal"):
                perm = get_scan_indices(
                    d, h, w, orient, x.device, self._perm_cache
                )
                inv = inverse_indices(perm)
                x_orient = xs[:, :, perm]
                y_orient = self._run_selective_scan(x_orient)
                outs.append(y_orient[:, :, inv])
            return sum(outs) / len(outs)

        perm = None
        if self.scan_mode in ("snake", "cross"):
            perm = get_scan_indices(d, h, w, self.scan_mode, x.device, self._perm_cache)
            xs = xs[:, :, perm]

        y = self._run_selective_scan(xs)

        if perm is not None:
            inv = inverse_indices(perm)
            y = y[:, :, inv]
        return y

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        b = x.shape[0]
        x = self.in_proj(x)
        if any(p > 0 for p in self._conv_pad):
            x = nn.functional.pad(x, self._conv_pad)
        x = self.act(self.conv3d(x))
        _, _, d, h, w = x.shape

        x = self.da_scan(x)
        y = self.ssm(x)
        y = y.reshape(b, self.d_inner, d, h, w)

        y = self.out_norm(y)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y
