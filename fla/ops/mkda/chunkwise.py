# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import torch


@torch.compiler.disable
def mkda_chunkwise_parallel(
    q: torch.Tensor,  # [B,T,H,K]
    k: torch.Tensor,  # [B,T,H,R,K]
    v: torch.Tensor,  # [B,T,H,R,V]
    log_alpha: torch.Tensor,  # [B,T,H,K] per-step
    beta: torch.Tensor,  # [B,T,H,R]
    *,
    initial_state: torch.Tensor | None = None,  # [B,H,K,V]
    scale: float | None = None,
    chunk_size: int = 64,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Chunkwise-parallel MKDA (Multi-Key Delta Attention).

    Implements a block-lower-triangular solve within each chunk (length C) and a cumsum,
    avoiding a Python loop over timesteps within the chunk. Chunks are processed sequentially
    via the carried recurrent state.

    All internal math is done in float32 for stability.
    """
    if q.ndim != 4:
        raise ValueError(f"Expected q to be [B,T,H,K], got {tuple(q.shape)}")
    if k.ndim != 5 or v.ndim != 5:
        raise ValueError(f"Expected k/v to be [B,T,H,R,*], got k={tuple(k.shape)} v={tuple(v.shape)}")
    if log_alpha.shape != q.shape:
        raise ValueError(f"Expected log_alpha same shape as q, got {tuple(log_alpha.shape)} vs {tuple(q.shape)}")
    if beta.ndim != 4:
        raise ValueError(f"Expected beta to be [B,T,H,R], got {tuple(beta.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    B, T, H, Kdim = q.shape
    R = k.shape[3]
    Vdim = v.shape[-1]
    device = q.device
    if scale is None:
        scale = Kdim**-0.5

    if initial_state is None:
        S0 = torch.zeros(B, H, Kdim, Vdim, device=device, dtype=torch.float32)
    else:
        S0 = initial_state.to(torch.float32).clone()

    # Important: torch.linalg.solve_triangular does not support fp16 on CUDA.
    # Also, autocast may downcast inputs to fp16 if enabled by the training loop.
    # We therefore run the entire chunkwise routine in a disabled-autocast region.
    device_type = q.device.type
    autocast_ctx = (
        torch.autocast(device_type=device_type, enabled=False)
        if hasattr(torch, "autocast")
        else torch.cuda.amp.autocast(enabled=False)
    )

    with autocast_ctx:
        q32 = q.to(torch.float32)
        k32 = k.to(torch.float32)
        v32 = v.to(torch.float32)
        la32 = log_alpha.to(torch.float32)
        b32 = beta.to(torch.float32)

        outs: list[torch.Tensor] = []

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            C = end - start

            q_c = q32[:, start:end]  # [B,C,H,K]
            k_c = k32[:, start:end]  # [B,C,H,R,K]
            v_c = v32[:, start:end]  # [B,C,H,R,V]
            la_c = la32[:, start:end]  # [B,C,H,K]
            be_c = b32[:, start:end]  # [B,C,H,R]

            # g_cum[t] = sum_{j<=t} log_alpha[j]  (within-chunk)
            g_cum = torch.cumsum(la_c, dim=1)  # [B,C,H,K]
            F = torch.exp(g_cum)  # [B,C,H,K]
            F_inv = torch.exp(-g_cum)  # [B,C,H,K]

            # K_mat: [B,C,H,K,R]
            K_mat = k_c.permute(0, 1, 2, 4, 3).contiguous()
            K_plus = K_mat * F.unsqueeze(-1)
            K_minus = K_mat * F_inv.unsqueeze(-1)
            K_minus_beta = K_minus * be_c.unsqueeze(-2)  # column-scale by beta

            # blocks[s,i] = K_plus[s]^T @ K_minus_beta[i]  => [B,H,C,C,R,R]
            blocks = torch.einsum("b s h k r, b i h k m -> b h s i r m", K_plus, K_minus_beta)
            tril = torch.tril(torch.ones(C, C, device=device, dtype=blocks.dtype), diagonal=-1)
            blocks = blocks * tril.view(1, 1, C, C, 1, 1)

            I_R = torch.eye(R, device=device, dtype=blocks.dtype)
            eye_C = torch.eye(C, device=device, dtype=blocks.dtype)
            Ablocks = blocks + eye_C.view(1, 1, C, C, 1, 1) * I_R.view(1, 1, 1, 1, R, R)

            # flatten: [B,H,C*R,C*R]
            Aflat = Ablocks.permute(0, 1, 2, 4, 3, 5).reshape(B, H, C * R, C * R)

            # RHS = V - K_plus^T S0
            pred0 = torch.einsum("b s h k r, b h k v -> b s h r v", K_plus, S0)
            rhs = v_c - pred0  # [B,C,H,R,V]
            rhsflat = rhs.permute(0, 2, 1, 3, 4).reshape(B, H, C * R, Vdim)

            # Force float32 for the solve (avoid any accidental downcast).
            Eflat = torch.linalg.solve_triangular(Aflat.float(), rhsflat.float(), upper=False, unitriangular=True)
            E = Eflat.reshape(B, H, C, R, Vdim).permute(0, 2, 1, 3, 4).contiguous()  # [B,C,H,R,V]

            # W_t = K_t * diag(beta_t) * E_t^T
            E_beta = E * be_c.unsqueeze(-1)
            W = torch.einsum("b s h k r, b s h r v -> b s h k v", K_mat, E_beta)  # [B,C,H,K,V]

            # U_t = cumsum(F_inv * W) and S_t = F * (S0 + U_t)
            U = torch.cumsum(W * F_inv.unsqueeze(-1), dim=1)
            S = (S0.unsqueeze(1) + U) * F.unsqueeze(-1)  # [B,C,H,K,V]

            o_c = torch.einsum("b s h k v, b s h k -> b s h v", S, q_c * float(scale))
            outs.append(o_c)

            S0 = S[:, -1]

        out = torch.cat(outs, dim=1) if outs else q32.new_zeros((B, 0, H, Vdim))
        return out, (S0 if output_final_state else None)
