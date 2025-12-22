# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import torch

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.gla.chunk import chunk_gla_fwd_intra_gk, chunk_gla_fwd_o_gk
from fla.ops.kda.chunk_intra_rank2 import (
    chunk_kda_rank2_bwd_mask_dAkk_within_token,
    chunk_kda_rank2_fwd_intra_a_inv,
)
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv, chunk_kda_bwd_dqkwg
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
from fla.ops.kda.wy_fast import recompute_w_u_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
from fla.ops.kda.wy_fast import prepare_wy_repr_bwd
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2


def _canonicalize_rank2_layout(
    x: torch.Tensor,
    *,
    name: str,
    expected_last_dim: int,
) -> torch.Tensor:
    """
    Accept [B,T,H,2,D] or [B,T,H,D,2] and return [B,T,H,2,D].
    """
    if x.ndim != 5:
        raise ValueError(f"Expected `{name}` to be 5D, got shape={tuple(x.shape)}.")
    if x.shape[-1] == expected_last_dim:
        # [B,T,H,2,D]
        return x
    if x.shape[-2] == expected_last_dim:
        # [B,T,H,D,2] -> [B,T,H,2,D]
        return x.transpose(-1, -2)
    raise ValueError(
        f"Failed to infer `{name}` layout. Expected last dim or second last dim to be {expected_last_dim}, "
        f"got shape={tuple(x.shape)}.",
    )


class ChunkKDARank2Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        log_alpha: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None,
        chunk_indices: torch.LongTensor | None,
    ):
        if q.ndim != 4:
            raise ValueError(f"Expected `q` to be [B,T,H,K], got shape={tuple(q.shape)}.")
        if log_alpha.shape != q.shape:
            raise ValueError(
                f"Expected `log_alpha` to match `q` shape [B,T,H,K], got {tuple(log_alpha.shape)} vs {tuple(q.shape)}.",
            )
        if beta.ndim != 4:
            raise ValueError(f"Expected `beta` to be [B,T,H,2], got shape={tuple(beta.shape)}.")

        B, T, H, K = q.shape
        if T == 0:
            V = v.shape[-1]
            o = q.new_empty((B, 0, H, V), dtype=v.dtype)
            final_state = None
            if output_final_state:
                if cu_seqlens is None:
                    N = B
                else:
                    N = len(cu_seqlens) - 1
                final_state = q.new_zeros((N, H, K, V), dtype=torch.float32)
            return o, final_state

        k_transposed = k.shape[-2] == K
        v_transposed = False

        k = _canonicalize_rank2_layout(k, name="k", expected_last_dim=K)
        if k.shape[:3] != (B, T, H) or k.shape[-2] != 2:
            raise ValueError(f"Expected `k` to be [B,T,H,2,K], got shape={tuple(k.shape)}.")
        Vdim = v.shape[-1] if v.shape[-1] != 2 else v.shape[-2]
        v_transposed = v.shape[-2] == Vdim
        v = _canonicalize_rank2_layout(v, name="v", expected_last_dim=Vdim)
        if v.shape[:4] != (B, T, H, 2):
            raise ValueError(f"Expected `v` to be [B,T,H,2,V], got shape={tuple(v.shape)}.")
        if beta.shape[:3] != (B, T, H) or beta.shape[-1] != 2:
            raise ValueError(f"Expected `beta` to be [B,T,H,2], got shape={tuple(beta.shape)}.")

        if scale is None:
            scale = K**-0.5

        initial_state_was_none = initial_state is None
        if cu_seqlens is not None:
            if q.shape[0] != 1:
                raise ValueError("When using cu_seqlens-packed inputs, expected B==1 (inputs flattened).")
            if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
                raise ValueError(
                    f"Expected initial_state.shape[0]=={len(cu_seqlens) - 1}, got {initial_state.shape[0]}."
                )

        if initial_state is None:
            N = B if cu_seqlens is None else (len(cu_seqlens) - 1)
            initial_state = q.new_zeros((N, H, K, Vdim), dtype=torch.float32)
        else:
            if initial_state.dtype != torch.float32:
                raise ValueError("Expected initial_state to be float32.")

        # Chunk-local cumsum in ln-space (for chunk_gla exp()).
        # We use a smaller token chunk (32) so that pseudo-time BT=64 remains compatible with existing bwd kernels.
        BT_TOK = 32
        BT_PSEUDO = 2 * BT_TOK
        chunk_indices_tok = chunk_indices
        if cu_seqlens is not None:
            # Always build indices for the token chunk size used here.
            chunk_indices_tok = prepare_chunk_indices(cu_seqlens, BT_TOK)
        g_ln = chunk_local_cumsum(
            g=log_alpha,
            chunk_size=BT_TOK,
            scale=1.0,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices_tok,
            output_dtype=torch.float32,
        )
        # Same cumsum but in log2-space (for exp2() kernels).
        g_log2 = g_ln * float(RCP_LN2)

        # Pseudo-time expansion (interleave rank into time).
        R = 2
        k_flat = k.permute(0, 1, 3, 2, 4).reshape(B, T * R, H, K)
        v_flat = v.permute(0, 1, 3, 2, 4).reshape(B, T * R, H, Vdim)
        beta_flat = beta.permute(0, 1, 3, 2).reshape(B, T * R, H)
        g_flat = g_log2.repeat_interleave(R, dim=1)
        g_ln_flat = g_ln.repeat_interleave(R, dim=1)

        # Queries exist only once per token; map to pseudo-time by placing q at odd indices (rank-1 slot).
        q_flat = torch.zeros((B, T * R, H, K), device=q.device, dtype=q.dtype)
        q_flat[:, 1::2] = q

        # Packed-varlen bookkeeping for pseudo-time.
        cu_seqlens2 = None if cu_seqlens is None else (cu_seqlens * R)
        chunk_indices2 = None
        if cu_seqlens2 is not None:
            chunk_indices2 = prepare_chunk_indices(cu_seqlens2, BT_PSEUDO)

        # Build intra-chunk inverse for the pseudo-time system, then compute (w,u,kg).
        Akk = chunk_kda_rank2_fwd_intra_a_inv(
            k=k_flat,
            gk=g_flat,
            beta=beta_flat,
            cu_seqlens=cu_seqlens2,
            chunk_indices=chunk_indices2,
            chunk_size=BT_PSEUDO,
        )
        w, u, _, kg = recompute_w_u_fwd(
            k=k_flat,
            v=v_flat,
            beta=beta_flat,
            A=Akk,
            q=None,
            gk=g_flat,
            cu_seqlens=cu_seqlens2,
            chunk_indices=chunk_indices2,
        )
        h, v_new_flat, final_state = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=g_flat,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens2,
            chunk_indices=chunk_indices2,
            chunk_size=BT_PSEUDO,
            use_exp2=True,
        )

        # Compute pseudo-time Aqk and output, then gather odd indices back to token time.
        A = chunk_gla_fwd_intra_gk(
            q=q_flat,
            k=k_flat,
            g=g_ln_flat,
            scale=float(scale),
            cu_seqlens=cu_seqlens2,
            chunk_size=BT_PSEUDO,
        )
        o_flat = chunk_gla_fwd_o_gk(
            q=q_flat,
            v=v_new_flat,
            g=g_flat,
            A=A,
            h=h,
            scale=float(scale),
            cu_seqlens=cu_seqlens2,
            chunk_size=BT_PSEUDO,
            chunk_indices=chunk_indices2,
            use_exp2=True,
        )
        o = o_flat[:, 1::2]

        ctx.save_for_backward(q, k, v, log_alpha, beta, initial_state)
        ctx.scale = float(scale)
        ctx.output_final_state = bool(output_final_state)
        ctx.cu_seqlens = cu_seqlens
        ctx.k_transposed = bool(k_transposed)
        ctx.v_transposed = bool(v_transposed)
        ctx.initial_state_was_none = initial_state_was_none
        return o, final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
        q, k, v, log_alpha, beta, initial_state = ctx.saved_tensors
        scale = float(ctx.scale)
        cu_seqlens = ctx.cu_seqlens
        initial_state_was_none = ctx.initial_state_was_none

        B, T, H, K = q.shape
        Vdim = v.shape[-1] if v.shape[-1] != 2 else v.shape[-2]
        k = _canonicalize_rank2_layout(k, name="k", expected_last_dim=K)
        v = _canonicalize_rank2_layout(v, name="v", expected_last_dim=Vdim)

        BT_TOK = 32
        BT_PSEUDO = 64

        chunk_indices_tok = None
        if cu_seqlens is not None:
            chunk_indices_tok = prepare_chunk_indices(cu_seqlens, BT_TOK)

        g_ln = chunk_local_cumsum(
            g=log_alpha,
            chunk_size=BT_TOK,
            scale=1.0,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices_tok,
            output_dtype=torch.float32,
        )
        g_log2 = g_ln * float(RCP_LN2)

        R = 2
        k_flat = k.permute(0, 1, 3, 2, 4).reshape(B, T * R, H, K)
        v_flat = v.permute(0, 1, 3, 2, 4).reshape(B, T * R, H, Vdim)
        beta_flat = beta.permute(0, 1, 3, 2).reshape(B, T * R, H)
        g_flat = g_log2.repeat_interleave(R, dim=1)
        g_ln_flat = g_ln.repeat_interleave(R, dim=1)

        q_flat = torch.zeros((B, T * R, H, K), device=q.device, dtype=q.dtype)
        q_flat[:, 1::2] = q

        cu_seqlens2 = None if cu_seqlens is None else (cu_seqlens * R)
        chunk_indices2 = None
        if cu_seqlens2 is not None:
            chunk_indices2 = prepare_chunk_indices(cu_seqlens2, BT_PSEUDO)

        Akk = chunk_kda_rank2_fwd_intra_a_inv(
            k=k_flat,
            gk=g_flat,
            beta=beta_flat,
            cu_seqlens=cu_seqlens2,
            chunk_indices=chunk_indices2,
            chunk_size=BT_PSEUDO,
        )
        w, u, qg, kg = recompute_w_u_fwd(
            k=k_flat,
            v=v_flat,
            beta=beta_flat,
            A=Akk,
            q=q_flat,
            gk=g_flat,
            cu_seqlens=cu_seqlens2,
            chunk_indices=chunk_indices2,
        )
        h, v_new_flat, _ = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=g_flat,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens2,
            chunk_indices=chunk_indices2,
            chunk_size=BT_PSEUDO,
            use_exp2=True,
        )

        A = chunk_gla_fwd_intra_gk(
            q=q_flat,
            k=k_flat,
            g=g_ln_flat,
            scale=float(scale),
            cu_seqlens=cu_seqlens2,
            chunk_size=BT_PSEUDO,
        )

        do_flat = torch.zeros((B, T * R, H, do.shape[-1]), device=do.device, dtype=do.dtype)
        do_flat[:, 1::2] = do

        dAqk, dv = chunk_kda_bwd_dAv(
            q=q_flat,
            k=k_flat,
            v=v_new_flat,
            do=do_flat,
            A=A,
            scale=float(scale),
            cu_seqlens=cu_seqlens2,
            chunk_size=BT_PSEUDO,
            chunk_indices=chunk_indices2,
        )
        dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
            q=qg,
            k=kg,
            w=w,
            gk=g_flat,
            h0=initial_state,
            dht=dht,
            do=do_flat,
            dv=dv,
            scale=float(scale),
            cu_seqlens=cu_seqlens2,
            chunk_size=BT_PSEUDO,
            chunk_indices=chunk_indices2,
            use_exp2=True,
        )
        dq, dk, dw, dg = chunk_kda_bwd_dqkwg(
            q=q_flat,
            k=k_flat,
            v=v_new_flat,
            w=w,
            g=g_flat,
            h=h,
            dv=dv,
            do=do_flat,
            dh=dh,
            scale=float(scale),
            cu_seqlens=cu_seqlens2,
            chunk_size=BT_PSEUDO,
            chunk_indices=chunk_indices2,
        )
        dk, dv, db, dg, dAkk = prepare_wy_repr_bwd(
            k=k_flat,
            v=v_flat,
            beta=beta_flat,
            gk=g_flat,
            A=Akk,
            dk=dk,
            dw=dw,
            du=dv,
            dg=dg,
            cu_seqlens=cu_seqlens2,
            chunk_indices=chunk_indices2,
        )
        chunk_kda_rank2_bwd_mask_dAkk_within_token(dAkk, cu_seqlens=cu_seqlens2, chunk_size=BT_PSEUDO)
        dq, dk, db, dg = chunk_kda_bwd_intra(
            q=q_flat,
            k=k_flat,
            g=g_flat,
            beta=beta_flat,
            dAqk=dAqk,
            dAkk=dAkk,
            dq=dq,
            dk=dk,
            db=db,
            dg=dg,
            cu_seqlens=cu_seqlens2,
            chunk_indices=chunk_indices2,
            chunk_size=BT_PSEUDO,
        )

        dq = dq[:, 1::2].to(q.dtype)
        dk = dk.reshape(B, T, R, H, K).permute(0, 1, 3, 2, 4).to(k.dtype)
        dv = dv.reshape(B, T, R, H, Vdim).permute(0, 1, 3, 2, 4).to(v.dtype)
        db = db.reshape(B, T, R, H).permute(0, 1, 3, 2).to(beta.dtype)
        dg = dg[:, 0::2].to(log_alpha.dtype)

        if ctx.k_transposed:
            dk = dk.transpose(-1, -2)
        if ctx.v_transposed:
            dv = dv.transpose(-1, -2)

        dh0 = None if initial_state_was_none else dh0
        return dq, dk, dv, dg, db, None, dh0, None, None, None


@torch.compiler.disable
def chunk_kda_rank2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Exact per-token rank-2 KDA/MKDA update, chunk-parallel forward in Triton.

    Shapes:
      q:         [B,T,H,K]
      k:         [B,T,H,2,K] (or [B,T,H,K,2])
      v:         [B,T,H,2,V] (or [B,T,H,V,2])
      log_alpha: [B,T,H,K]   (per-step log decay, ln-space; <=0)
      beta:      [B,T,H,2]
      initial_state: [N,H,K,V] float32, with N=B (fixed-len) or N=len(cu_seqlens)-1 (varlen)

    Returns:
      o:  [B,T,H,V]
      ht: [N,H,K,V] float32 if output_final_state else None
    """
    return ChunkKDARank2Function.apply(
        q,
        k,
        v,
        log_alpha,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
    )
