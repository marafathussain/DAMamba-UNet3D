import torch
from einops import rearrange

try:
    import selective_scan_cuda_oflex_rh
except Exception:  # pragma: no cover - extension may be built in-package
    from . import selective_scan_cuda_oflex_rh


def flops_selective_scan_fn(
    B: int = 1,
    L: int = 256,
    D: int = 768,
    N: int = 16,
    with_C: bool = True,
    with_D: bool = True,
    with_Z: bool = False,
    with_complex: bool = False,
):
    assert not with_complex
    if with_C:
        flops = 9 * B * L * D * N
    else:
        flops = 7 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def selective_scan_state_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn):
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    assert N == 1
    flops = flops_fn(B=B, L=L, D=D, N=N, with_C=False, with_D=False, with_Z=False)
    return flops


class SelectiveScanStateFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u,
        delta,
        A,
        B,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
        lag=0,
    ):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True

        out, x, *rest = selective_scan_cuda_oflex_rh.fwd(
            u, delta, A, B, D, delta_bias, delta_softplus, 1, True
        )
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        ctx.save_for_backward(u, delta, A, B, D, z, delta_bias, x, out)
        out_z = rest[0]
        return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dD, ddelta_bias, *rest = selective_scan_cuda_oflex_rh.bwd(
            u, delta, A, B, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        return (
            du,
            ddelta,
            dA,
            dB,
            dD if D is not None else None,
            dz,
            ddelta_bias if delta_bias is not None else None,
            None,
            None,
            None,
        )


def selective_scan_fn(
    u,
    delta,
    A,
    B,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    return SelectiveScanStateFn.apply(
        u, delta, A, B, D, z, delta_bias, delta_softplus, return_last_state
    )
