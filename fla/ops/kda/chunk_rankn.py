# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import torch

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.gla.chunk import chunk_gla_fwd_intra_gk, chunk_gla_fwd_o_gk
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv, chunk_kda_bwd_dqkwg
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
from fla.ops.kda.chunk_intra_rank4 import chunk_kda_rank4_bwd_mask_dAkk_within_token, chunk_kda_rank4_fwd_intra_a_inv
from fla.ops.kda.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2


def _canonicalize_rank_layout(
    x: torch.Tensor,
    *,
    name: str,
    expected_rank: int,
    expected_feat_dim: int,
) -> tuple[torch.Tensor, bool]:
    """
    Accept [B,T,H,R,D] or [B,T,H,D,R] and return ([B,T,H,R,D], transposed_flag).
    """
    if x.ndim != 5:
        raise ValueError(f"Expected `{name}` to be 5D, got shape={tuple(x.shape)}.")

    if x.shape[-1] == expected_feat_dim:
        if x.shape[-2] != expected_rank:
            raise ValueError(
                f"Expected `{name}` to be [B,T,H,{expected_rank},{expected_feat_dim}], got shape={tuple(x.shape)}.",
            )
        return x, False
    if x.shape[-2] == expected_feat_dim:
        if x.shape[-1] != expected_rank:
            raise ValueError(
                f"Expected `{name}` to be [B,T,H,{expected_feat_dim},{expected_rank}], got shape={tuple(x.shape)}.",
            )
        return x.transpose(-1, -2), True

    raise ValueError(
        f"Failed to infer `{name}` layout. Expected last dim or second last dim to be {expected_feat_dim}, "
        f"got shape={tuple(x.shape)}.",
    )


class ChunkKDARankNFunction(torch.autograd.Function):
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
            raise ValueError(f"Expected `beta` to be [B,T,H,R], got shape={tuple(beta.shape)}.")

        B, T, H, K = q.shape
        R_in = int(beta.shape[-1])
        if R_in <= 0 or R_in > 4:
            raise ValueError(f"Expected rank R in [1,4], got R={R_in}.")

        if v.ndim != 5:
            raise ValueError(f"Expected `v` to be [B,T,H,R,V] (or [B,T,H,V,R]), got shape={tuple(v.shape)}.")
        if v.shape[-2] == R_in:
            Vdim = int(v.shape[-1])
        elif v.shape[-1] == R_in:
            Vdim = int(v.shape[-2])
        else:
            raise ValueError(
                f"Failed to infer `v` layout with rank={R_in}; expected v.shape[-2]==R or v.shape[-1]==R, "
                f"got shape={tuple(v.shape)}.",
            )

        if T == 0:
            o = q.new_empty((B, 0, H, Vdim), dtype=v.dtype)
            final_state = None
            if output_final_state:
                N = B if cu_seqlens is None else (len(cu_seqlens) - 1)
                final_state = q.new_zeros((N, H, K, Vdim), dtype=torch.float32)
            return o, final_state

        if scale is None:
            scale = K**-0.5

        if cu_seqlens is not None:
            if q.shape[0] != 1:
                raise ValueError("When using cu_seqlens-packed inputs, expected B==1 (inputs flattened).")
            if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
                raise ValueError(
                    f"Expected initial_state.shape[0]=={len(cu_seqlens) - 1}, got {initial_state.shape[0]}.",
                )

        initial_state_was_none = initial_state is None
        if initial_state is None:
            N = B if cu_seqlens is None else (len(cu_seqlens) - 1)
            # delta-rule kernels expect float32 state
            initial_state = q.new_zeros((N, H, K, Vdim), dtype=torch.float32)
        else:
            if initial_state.dtype != torch.float32:
                raise ValueError("Expected initial_state to be float32.")
            expected_n = B if cu_seqlens is None else (len(cu_seqlens) - 1)
            if initial_state.shape != (expected_n, H, K, Vdim):
                raise ValueError(
                    f"Expected initial_state shape {(expected_n, H, K, Vdim)}, got shape={tuple(initial_state.shape)}.",
                )

        k, k_transposed = _canonicalize_rank_layout(k, name="k", expected_rank=R_in, expected_feat_dim=K)
        v, v_transposed = _canonicalize_rank_layout(v, name="v", expected_rank=R_in, expected_feat_dim=Vdim)
        if k.shape[:3] != (B, T, H):
            raise ValueError(f"Expected `k` leading dims to be [B,T,H], got shape={tuple(k.shape)}.")
        if v.shape[:3] != (B, T, H):
            raise ValueError(f"Expected `v` leading dims to be [B,T,H], got shape={tuple(v.shape)}.")
        if beta.shape[:3] != (B, T, H):
            raise ValueError(f"Expected `beta` leading dims to be [B,T,H], got shape={tuple(beta.shape)}.")

        # Chunk-local cumsum in ln-space (for chunk_gla exp()).
        BT_TOK = 32
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

        # Pad rank to R_MAX=4 and expand to pseudo-time (interleave rank into time).
        R_MAX = 4
        if R_in == R_MAX:
            k4, v4, beta4 = k, v, beta
        else:
            k4 = torch.zeros((B, T, H, R_MAX, K), device=k.device, dtype=k.dtype)
            v4 = torch.zeros((B, T, H, R_MAX, Vdim), device=v.device, dtype=v.dtype)
            beta4 = torch.zeros((B, T, H, R_MAX), device=beta.device, dtype=beta.dtype)
            k4[..., :R_in, :] = k
            v4[..., :R_in, :] = v
            beta4[..., :R_in] = beta

        k_flat = k4.permute(0, 1, 3, 2, 4).reshape(B, T * R_MAX, H, K)
        v_flat = v4.permute(0, 1, 3, 2, 4).reshape(B, T * R_MAX, H, Vdim)
        beta_flat = beta4.permute(0, 1, 3, 2).reshape(B, T * R_MAX, H)
        g_flat = g_log2.repeat_interleave(R_MAX, dim=1)
        g_ln_flat = g_ln.repeat_interleave(R_MAX, dim=1)

        # Queries exist only once per token; map to pseudo-time by placing q at the last pseudo step.
        q_flat = torch.zeros((B, T * R_MAX, H, K), device=q.device, dtype=q.dtype)
        q_flat[:, (R_MAX - 1) :: R_MAX] = q

        BT_PSEUDO = 128
        cu_seqlens_p = None if cu_seqlens is None else (cu_seqlens * R_MAX)
        chunk_indices_p = None
        if cu_seqlens_p is not None:
            chunk_indices_p = prepare_chunk_indices(cu_seqlens_p, BT_PSEUDO)

        Akk = chunk_kda_rank4_fwd_intra_a_inv(
            k=k_flat,
            gk=g_flat,
            beta=beta_flat,
            cu_seqlens=cu_seqlens_p,
            chunk_indices=chunk_indices_p,
            chunk_size=BT_PSEUDO,
        )
        w, u, _, kg = recompute_w_u_fwd(
            k=k_flat,
            v=v_flat,
            beta=beta_flat,
            A=Akk,
            q=None,
            gk=g_flat,
            cu_seqlens=cu_seqlens_p,
            chunk_indices=chunk_indices_p,
        )
        h, v_new_flat, final_state = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=g_flat,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens_p,
            chunk_indices=chunk_indices_p,
            chunk_size=BT_PSEUDO,
            use_exp2=True,
        )

        A = chunk_gla_fwd_intra_gk(
            q=q_flat,
            k=k_flat,
            g=g_ln_flat,
            scale=float(scale),
            cu_seqlens=cu_seqlens_p,
            chunk_size=BT_PSEUDO,
        )
        o_flat = chunk_gla_fwd_o_gk(
            q=q_flat,
            v=v_new_flat,
            g=g_flat,
            A=A,
            h=h,
            scale=float(scale),
            cu_seqlens=cu_seqlens_p,
            chunk_size=BT_PSEUDO,
            chunk_indices=chunk_indices_p,
            use_exp2=True,
        )
        o = o_flat[:, (R_MAX - 1) :: R_MAX]

        ctx.save_for_backward(q, k, v, log_alpha, beta, initial_state)
        ctx.scale = float(scale)
        ctx.output_final_state = bool(output_final_state)
        ctx.cu_seqlens = cu_seqlens
        ctx.k_transposed = bool(k_transposed)
        ctx.v_transposed = bool(v_transposed)
        ctx.v_dim = int(Vdim)
        ctx.rank_in = int(R_in)
        ctx.initial_state_was_none = bool(initial_state_was_none)
        return o, final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
        q, k, v, log_alpha, beta, initial_state = ctx.saved_tensors
        scale = float(ctx.scale)
        cu_seqlens = ctx.cu_seqlens
        Vdim = int(ctx.v_dim)
        R_in = int(ctx.rank_in)
        initial_state_was_none = bool(ctx.initial_state_was_none)

        B, T, H, K = q.shape

        k, _ = _canonicalize_rank_layout(k, name="k", expected_rank=R_in, expected_feat_dim=K)
        v, _ = _canonicalize_rank_layout(v, name="v", expected_rank=R_in, expected_feat_dim=Vdim)

        BT_TOK = 32
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

        R_MAX = 4
        if R_in == R_MAX:
            k4, v4, beta4 = k, v, beta
        else:
            k4 = torch.zeros((B, T, H, R_MAX, K), device=k.device, dtype=k.dtype)
            v4 = torch.zeros((B, T, H, R_MAX, Vdim), device=v.device, dtype=v.dtype)
            beta4 = torch.zeros((B, T, H, R_MAX), device=beta.device, dtype=beta.dtype)
            k4[..., :R_in, :] = k
            v4[..., :R_in, :] = v
            beta4[..., :R_in] = beta

        k_flat = k4.permute(0, 1, 3, 2, 4).reshape(B, T * R_MAX, H, K)
        v_flat = v4.permute(0, 1, 3, 2, 4).reshape(B, T * R_MAX, H, Vdim)
        beta_flat = beta4.permute(0, 1, 3, 2).reshape(B, T * R_MAX, H)
        g_flat = g_log2.repeat_interleave(R_MAX, dim=1)
        g_ln_flat = g_ln.repeat_interleave(R_MAX, dim=1)

        q_flat = torch.zeros((B, T * R_MAX, H, K), device=q.device, dtype=q.dtype)
        q_flat[:, (R_MAX - 1) :: R_MAX] = q

        BT_PSEUDO = 128
        cu_seqlens_p = None if cu_seqlens is None else (cu_seqlens * R_MAX)
        chunk_indices_p = None
        if cu_seqlens_p is not None:
            chunk_indices_p = prepare_chunk_indices(cu_seqlens_p, BT_PSEUDO)

        Akk = chunk_kda_rank4_fwd_intra_a_inv(
            k=k_flat,
            gk=g_flat,
            beta=beta_flat,
            cu_seqlens=cu_seqlens_p,
            chunk_indices=chunk_indices_p,
            chunk_size=BT_PSEUDO,
        )
        w, u, qg, kg = recompute_w_u_fwd(
            k=k_flat,
            v=v_flat,
            beta=beta_flat,
            A=Akk,
            q=q_flat,
            gk=g_flat,
            cu_seqlens=cu_seqlens_p,
            chunk_indices=chunk_indices_p,
        )
        h, v_new_flat, _ = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=g_flat,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens_p,
            chunk_indices=chunk_indices_p,
            chunk_size=BT_PSEUDO,
            use_exp2=True,
        )
        A = chunk_gla_fwd_intra_gk(
            q=q_flat,
            k=k_flat,
            g=g_ln_flat,
            scale=float(scale),
            cu_seqlens=cu_seqlens_p,
            chunk_size=BT_PSEUDO,
        )

        do_flat = torch.zeros((B, T * R_MAX, H, do.shape[-1]), device=do.device, dtype=do.dtype)
        do_flat[:, (R_MAX - 1) :: R_MAX] = do

        dAqk, dv = chunk_kda_bwd_dAv(
            q=q_flat,
            k=k_flat,
            v=v_new_flat,
            do=do_flat,
            A=A,
            scale=float(scale),
            cu_seqlens=cu_seqlens_p,
            chunk_size=BT_PSEUDO,
            chunk_indices=chunk_indices_p,
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
            cu_seqlens=cu_seqlens_p,
            chunk_size=BT_PSEUDO,
            chunk_indices=chunk_indices_p,
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
            cu_seqlens=cu_seqlens_p,
            chunk_size=BT_PSEUDO,
            chunk_indices=chunk_indices_p,
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
            cu_seqlens=cu_seqlens_p,
            chunk_indices=chunk_indices_p,
        )
        chunk_kda_rank4_bwd_mask_dAkk_within_token(dAkk, cu_seqlens=cu_seqlens_p, chunk_size=BT_PSEUDO)
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
            cu_seqlens=cu_seqlens_p,
            chunk_indices=chunk_indices_p,
            chunk_size=BT_PSEUDO,
        )

        dq = dq[:, (R_MAX - 1) :: R_MAX].to(q.dtype)
        dk = dk.reshape(B, T, R_MAX, H, K).permute(0, 1, 3, 2, 4)[..., :R_in, :].to(k.dtype)
        dv = dv.reshape(B, T, R_MAX, H, Vdim).permute(0, 1, 3, 2, 4)[..., :R_in, :].to(v.dtype)
        db = db.reshape(B, T, R_MAX, H).permute(0, 1, 3, 2)[..., :R_in].to(beta.dtype)
        dg = dg[:, 0::R_MAX].to(log_alpha.dtype)

        if ctx.k_transposed:
            dk = dk.transpose(-1, -2)
        if ctx.v_transposed:
            dv = dv.transpose(-1, -2)

        dh0 = None if initial_state_was_none else dh0
        return dq, dk, dv, dg, db, None, dh0, None, None, None


@torch.compiler.disable
def chunk_kda_rank_n(
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
    Exact per-token rank-R KDA/MKDA update for R<=4, chunk-parallel forward/backward in Triton.

    Shapes:
      q:         [B,T,H,K]
      k:         [B,T,H,R,K] (or [B,T,H,K,R])
      v:         [B,T,H,R,V] (or [B,T,H,V,R])
      log_alpha: [B,T,H,K]   (per-token log decay, ln-space; <=0)
      beta:      [B,T,H,R]
      initial_state: [N,H,K,V] float32, with N=B (fixed-len) or N=len(cu_seqlens)-1 (varlen)

    Returns:
      o:  [B,T,H,V]
      ht: [N,H,K,V] float32 if output_final_state else None
    """
    return ChunkKDARankNFunction.apply(
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
