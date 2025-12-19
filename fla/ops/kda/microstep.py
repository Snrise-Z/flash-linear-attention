# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import torch

from fla.ops.kda.chunk import chunk_kda
from fla.ops.kda.fused_recurrent import fused_recurrent_kda


def _canonicalize_rank_r_layout(
    x: torch.Tensor,
    *,
    name: str,
    expected_last_dim: int,
) -> torch.Tensor:
    """
    Accepts either [..., R, D] or [..., D, R] and returns [..., R, D].
    """
    if x.ndim != 5:
        raise ValueError(f"Expected `{name}` to be 5D, got shape={tuple(x.shape)}.")

    if x.shape[-1] == expected_last_dim:
        # [..., R, D]
        return x
    if x.shape[-2] == expected_last_dim:
        # [..., D, R] -> [..., R, D]
        return x.transpose(-1, -2)

    raise ValueError(
        f"Failed to infer `{name}` layout. Expected last dim or second last dim to be {expected_last_dim}, "
        f"got shape={tuple(x.shape)}.",
    )


def _expand_to_microsteps(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    use_gate_in_kernel: bool,
    cu_seqlens: torch.LongTensor | None,
    fill_g_raw: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor | None, int]:
    """
    Expand rank-r inputs into a rank-1 micro-step sequence of length T*r.

    Shapes (fixed-length):
      q:    [B, T, H, K]
      k:    [B, T, H, R, K]
      v:    [B, T, H, R, V]
      g:    [B, T, H, K]
      beta: [B, T, H, R]

    Returns:
      q_micro:    [B, T*R, H, K]
      k_micro:    [B, T*R, H, K]
      v_micro:    [B, T*R, H, V]
      g_micro:    [B, T*R, H, K]
      beta_micro: [B, T*R, H]
      cu_seqlens_micro: [N+1] or None
      R: rank
    """
    if q.ndim != 4:
        raise ValueError(f"Expected `q` to be 4D [B,T,H,K], got shape={tuple(q.shape)}.")
    if g.shape != q.shape:
        raise ValueError(f"Expected `g` to have same shape as `q`, got g={tuple(g.shape)} q={tuple(q.shape)}.")
    if k.ndim != 5:
        raise ValueError(f"Expected `k` to be 5D [B,T,H,R,K], got shape={tuple(k.shape)}.")
    if v.ndim != 5:
        raise ValueError(f"Expected `v` to be 5D [B,T,H,R,V], got shape={tuple(v.shape)}.")
    if beta.ndim != 4:
        raise ValueError(f"Expected `beta` to be 4D [B,T,H,R], got shape={tuple(beta.shape)}.")

    B, T, H, K = q.shape
    if k.shape[:3] != (B, T, H):
        raise ValueError(f"Expected `k` leading dims to be (B,T,H)={B,T,H}, got shape={tuple(k.shape)}.")
    if v.shape[:3] != (B, T, H):
        raise ValueError(f"Expected `v` leading dims to be (B,T,H)={B,T,H}, got shape={tuple(v.shape)}.")
    if beta.shape[:3] != (B, T, H):
        raise ValueError(f"Expected `beta` leading dims to be (B,T,H)={B,T,H}, got shape={tuple(beta.shape)}.")

    if k.shape[-1] != K and k.shape[-2] != K:
        raise ValueError(
            f"Expected `k` last or second-last dim to match K={K}, got shape={tuple(k.shape)}.",
        )
    k = _canonicalize_rank_r_layout(k, name="k", expected_last_dim=K)  # [B,T,H,R,K]

    R = k.shape[-2]
    if R <= 0:
        raise ValueError(f"Expected rank R>0, got R={R}.")

    V = v.shape[-1] if v.shape[-1] != R else v.shape[-2]
    v = _canonicalize_rank_r_layout(v, name="v", expected_last_dim=V)  # [B,T,H,R,V]

    if v.shape[-2] != R:
        raise ValueError(f"Expected `v` rank dim R={R} to match `k`, got v shape={tuple(v.shape)}.")
    if beta.shape[-1] != R:
        raise ValueError(f"Expected `beta` last dim R={R}, got beta shape={tuple(beta.shape)}.")

    # Expand q: repeat each token R times, but only keep q on the last micro-step.
    # This matches the "emit output only after the last micro-step" convention and avoids
    # spending work on intermediate micro-step outputs.
    q_micro = q.repeat_interleave(R, dim=1)  # [B, T*R, H, K]
    if R > 1:
        micro_rank = torch.arange(T * R, device=q.device) % R
        is_last = micro_rank == (R - 1)
        q_micro = torch.where(is_last.view(1, -1, 1, 1), q_micro, q.new_zeros(()))

    # Expand k,v,beta: move rank to time axis.
    k_micro = k.permute(0, 1, 3, 2, 4).reshape(B, T * R, H, K)
    v_micro = v.permute(0, 1, 3, 2, 4).reshape(B, T * R, H, V)
    beta_micro = beta.permute(0, 1, 3, 2).reshape(B, T * R, H)

    # Expand g: apply decay only on the first micro-step per token.
    g_rep = g.repeat_interleave(R, dim=1)
    if R == 1:
        g_micro = g_rep
    else:
        micro_rank = torch.arange(T * R, device=g.device) % R
        is_first = micro_rank == 0
        if use_gate_in_kernel:
            fill = torch.tensor(fill_g_raw, dtype=g.dtype, device=g.device)
        else:
            fill = torch.tensor(0.0, dtype=g.dtype, device=g.device)
        g_micro = torch.where(is_first.view(1, -1, 1, 1), g_rep, fill)

    cu_seqlens_micro = None if cu_seqlens is None else (cu_seqlens * R)
    return q_micro, k_micro, v_micro, g_micro, beta_micro, cu_seqlens_micro, R


@torch.compiler.disable
def chunk_kda_rank_r_microstep(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    fill_g_raw: float = -1.0e4,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Approximation B for rank-r KDA: treat each token as `r` serial rank-1 micro-steps.

    Conceptually, for each original token t, we apply:
      1) One decay step using `g_t`
      2) `r` sequential rank-1 updates using (k_{t,a}, v_{t,a}, beta_{t,a})
      3) Emit output only after the last micro-step (this function returns length-T outputs)

    This wrapper implements the approximation by expanding the sequence length from `T` to `T*r`,
    then calling the existing rank-1 `chunk_kda` kernel.

    Expected shapes:
      q:    [B, T, H, K]
      g:    [B, T, H, K]  (raw gate if `use_gate_in_kernel=True`, else log-decay)
      k:    [B, T, H, R, K] (or [B, T, H, K, R])
      v:    [B, T, H, R, V] (or [B, T, H, V, R])
      beta: [B, T, H, R]

    Notes:
      - Decay is applied on the first micro-step per token; subsequent micro-steps use no decay.
      - If `use_gate_in_kernel=True`, we set the raw gate for non-first micro-steps to `fill_g_raw`
        (a large negative number), making the computed decay approximately 0.
      - `chunk_indices` is not supported because the sequence length changes; pass `None`.
    """
    if chunk_indices is not None:
        raise ValueError("`chunk_indices` is not supported for micro-step expansion; pass `chunk_indices=None`.")

    q_micro, k_micro, v_micro, g_micro, beta_micro, cu_seqlens_micro, R = _expand_to_microsteps(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        use_gate_in_kernel=use_gate_in_kernel,
        cu_seqlens=cu_seqlens,
        fill_g_raw=fill_g_raw,
    )

    o_micro, final_state = chunk_kda(
        q=q_micro,
        k=k_micro,
        v=v_micro,
        g=g_micro,
        beta=beta_micro,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        cu_seqlens=cu_seqlens_micro,
        chunk_indices=None,
        **kwargs,
    )

    # Take only the last micro-step output per original token.
    B, TR, H, V = o_micro.shape
    if TR % R != 0:
        raise RuntimeError(f"Internal error: TR={TR} is not divisible by R={R}.")
    T = TR // R
    o = o_micro.reshape(B, T, R, H, V)[:, :, -1]
    return o, final_state


@torch.compiler.disable
def fused_recurrent_kda_rank_r_microstep(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    fill_g_raw: float = -1.0e4,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Micro-step approximation (same as `chunk_kda_rank_r_microstep`) but using `fused_recurrent_kda`.

    Important: unlike `chunk_kda`, `fused_recurrent_kda` expects `g` to be the *log-decay* already
    (i.e., after `fused_kda_gate`). Therefore `use_gate_in_kernel` is not supported here.

    Shapes:
      q:    [B, T, H, K]
      g:    [B, T, H, K]  (log-decay)
      k:    [B, T, H, R, K] (or [B, T, H, K, R])
      v:    [B, T, H, R, V] (or [B, T, H, V, R])
      beta: [B, T, H, R]
    """
    # Reuse the same expander but force "decay g" path by passing use_gate_in_kernel=False.
    q_micro, k_micro, v_micro, g_micro, beta_micro, cu_seqlens_micro, R = _expand_to_microsteps(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        use_gate_in_kernel=False,
        cu_seqlens=cu_seqlens,
        fill_g_raw=fill_g_raw,
    )

    o_micro, final_state = fused_recurrent_kda(
        q=q_micro,
        k=k_micro,
        v=v_micro,
        g=g_micro,
        beta=beta_micro,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens_micro,
        **kwargs,
    )

    B, TR, H, V = o_micro.shape
    if TR % R != 0:
        raise RuntimeError(f"Internal error: TR={TR} is not divisible by R={R}.")
    T = TR // R
    o = o_micro.reshape(B, T, R, H, V)[:, :, -1]
    return o, final_state
