# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import torch


@torch.compiler.disable
def mkda_recurrent(
    q: torch.Tensor,  # [B,T,H,K]
    k: torch.Tensor,  # [B,T,H,R,K]
    v: torch.Tensor,  # [B,T,H,R,V]
    log_alpha: torch.Tensor,  # [B,T,H,K] per-step
    beta: torch.Tensor,  # [B,T,H,R]
    *,
    initial_state: torch.Tensor | None = None,  # [B,H,K,V]
    scale: float | None = None,
    output_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Reference recurrent MKDA (Multi-Key Delta Attention).

    This is a correctness-oriented implementation intended for tests/debugging.
    """
    if q.ndim != 4:
        raise ValueError(f"Expected q to be [B,T,H,K], got {tuple(q.shape)}")
    if k.ndim != 5 or v.ndim != 5:
        raise ValueError(f"Expected k/v to be [B,T,H,R,*], got k={tuple(k.shape)} v={tuple(v.shape)}")
    if log_alpha.shape != q.shape:
        raise ValueError(f"Expected log_alpha same shape as q, got {tuple(log_alpha.shape)} vs {tuple(q.shape)}")
    if beta.ndim != 4:
        raise ValueError(f"Expected beta to be [B,T,H,R], got {tuple(beta.shape)}")

    B, T, H, Kdim = q.shape
    R = k.shape[3]
    Vdim = v.shape[-1]
    if scale is None:
        scale = Kdim**-0.5

    device_type = q.device.type
    autocast_ctx = (
        torch.autocast(device_type=device_type, enabled=False)
        if hasattr(torch, "autocast")
        else torch.cuda.amp.autocast(enabled=False)
    )

    with autocast_ctx:
        if initial_state is None:
            S = torch.zeros(B, H, Kdim, Vdim, device=q.device, dtype=torch.float32)
        else:
            S = initial_state.to(torch.float32).clone()

        out = torch.empty(B, T, H, Vdim, device=q.device, dtype=torch.float32)

        q32 = q.to(torch.float32)
        k32 = k.to(torch.float32)
        v32 = v.to(torch.float32)
        la32 = log_alpha.to(torch.float32)
        b32 = beta.to(torch.float32)

        for t in range(T):
            alpha = la32[:, t].exp()  # [B,H,K]
            S = S * alpha.unsqueeze(-1)

            kt = k32[:, t]  # [B,H,R,K]
            vt = v32[:, t]  # [B,H,R,V]
            bt = b32[:, t]  # [B,H,R]

            pred = torch.einsum("b h k v, b h r k -> b h r v", S, kt)
            resid = (vt - pred) * bt.unsqueeze(-1)
            dS = torch.einsum("b h r k, b h r v -> b h k v", kt, resid)
            S = S + dS

            qt = q32[:, t] * float(scale)
            out[:, t] = torch.einsum("b h k v, b h k -> b h v", S, qt)

        return out, (S if output_final_state else None)
