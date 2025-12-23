# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, solve_tril
from fla.ops.utils.op import exp2
from fla.utils import IS_TF32_SUPPORTED, autotune_cache_kwargs

if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32x3")
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BH": BH}, num_warps=num_warps)
        for BH in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["K", "H"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "N"])
def chunk_kda_rank2_fwd_kernel_intra_token_parallel(
    k,
    g,
    beta,
    Akkd,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Compute diagonal BCxBC blocks (strictly lower) of the pseudo-time Akk (L) matrix.

    Inputs are already expanded to pseudo-time (T = 2 * T_tokens):
      - k:    [B, T, H, K]
      - g:    [B, T, H, K]   (chunk-local cumsum, log2)
      - beta: [B, T, H]

    We zero within-token couplings by enforcing: L[i,j]=0 when floor(i/2)==floor(j/2).
    """
    i_tg, i_hg = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = 0
        left, right = 0, N
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left

        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T

    if i_t >= T:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC

    k += bos * H * K
    g += bos * H * K
    beta += bos * H
    Akkd += bos * H * BC

    BK: tl.constexpr = triton.next_power_of_2(K)
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, BK)
    m_h = (i_hg * BH + o_h) < H
    m_k = o_k < K

    # Row i (pseudo-time)
    p_ki = tl.make_block_ptr(k + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_gi = tl.make_block_ptr(g + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_bi = tl.make_block_ptr(beta + i_t * H, (H,), (1,), (i_hg * BH,), (BH,), (0,))

    b_k = tl.load(p_ki, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_gi, boundary_check=(0, 1)).to(tl.float32)
    b_b = tl.load(p_bi, boundary_check=(0,)).to(tl.float32)
    b_k = b_k * b_b[:, None]

    token_start = (i_t // 2) * 2
    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        p_kj = tl.make_block_ptr(k + j * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        p_gj = tl.make_block_ptr(g + j * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        b_kj = tl.load(p_kj, boundary_check=(0, 1)).to(tl.float32)
        b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)

        b_kgj = tl.where(m_k[None, :], b_kj * exp2(b_g - b_gj), 0.0)
        b_Akk = tl.where(j < token_start, tl.sum(b_k * b_kgj, axis=1), 0.0)

        tl.store(
            Akkd + i_t * H * BC + (i_hg * BH + o_h) * BC + (j % BC),
            b_Akk.to(Akkd.dtype.element_ty),
            mask=m_h,
        )


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_rank2_fwd_kernel_inter_solve_fused(
    k,
    g,
    beta,
    Akkd,
    Akk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel: compute off-diagonal Akk blocks for a BT=128 pseudo-time chunk and
    invert the 4x4 block unit-lower-triangular system (each block is BCxBC, BC=32).

    Prerequisite: `chunk_kda_rank2_fwd_kernel_intra_token_parallel` has written the diagonal
    BCxBC blocks into Akkd (fp32, strictly lower; diagonal/upper ignored).
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    beta += bos * H + i_h
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_g0 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        b_k1, b_g1 = b_k0, b_g0
        b_k2, b_g2 = b_k0, b_g0
        if i_tc1 < T:
            p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)

            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn1 = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            b_Akk10 += tl.dot(b_k1 * b_gqn1, b_kgt)

        if i_tc2 < T:
            p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_g2 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
            b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)

            b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
            b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
            b_Akk20 += tl.dot(b_k2 * b_gqn2, b_kgt)
            b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
            b_Akk21 += tl.dot(b_k2 * b_gqn2, b_kgt)

        if i_tc3 < T:
            p_k3 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            p_g3 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
            b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)

            b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
            b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
            b_Akk30 += tl.dot(b_k3 * b_gqn3, b_kgt)
            b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
            b_Akk31 += tl.dot(b_k3 * b_gqn3, b_kgt)
            b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
            b_Akk32 += tl.dot(b_k3 * b_gqn3, b_kgt)

    # Apply beta(row) scaling to off-diagonal blocks.
    if i_tc1 < T:
        p_b1 = tl.make_block_ptr(beta, (T,), (H,), (i_tc1,), (BC,), (0,))
        b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
        b_Akk10 *= b_b1[:, None]
    if i_tc2 < T:
        p_b2 = tl.make_block_ptr(beta, (T,), (H,), (i_tc2,), (BC,), (0,))
        b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
        b_Akk20 *= b_b2[:, None]
        b_Akk21 *= b_b2[:, None]
    if i_tc3 < T:
        p_b3 = tl.make_block_ptr(beta, (T,), (H,), (i_tc3,), (BC,), (0,))
        b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
        b_Akk30 *= b_b3[:, None]
        b_Akk31 *= b_b3[:, None]
        b_Akk32 *= b_b3[:, None]

    # Load diagonal blocks (strictly lower L values) from Akkd.
    p_Akk00 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    # Invert diagonal unit-lower blocks via forward substitution.
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Ai00 = -tl.where(m_A, b_Ai00, 0)
    b_Ai11 = -tl.where(m_A, b_Ai11, 0)
    b_Ai22 = -tl.where(m_A, b_Ai22, 0)
    b_Ai33 = -tl.where(m_A, b_Ai33, 0)

    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
        b_a00 = tl.where(o_i < i, b_a00, 0.0)
        b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(BC + 2, min(2 * BC, T - i_tc0)):
        b_a11 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
        b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
        b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
    for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
        b_a22 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
        b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
        b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
    for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
        b_a33 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
        b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
        b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    # Merge inverse with off-diagonals.
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    # Store full inverse blocks to Akk.
    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk21 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
    p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_Akk31 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_Akk32 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["H", "K", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_rank2_fwd_kernel_build_L(
    k,
    g,
    beta,
    L,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NB: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Build the strictly-lower pseudo-time L matrix (within-token couplings masked) for rank-2 pseudo-time.

    Within-token masking: L[i, j] = 0 when floor(i/2) == floor(j/2), which is equivalent to j < (i//2)*2.
    """
    i_t, i_blk_pair, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    if i_t * BT >= T:
        return

    i_blk = i_blk_pair // NB
    j_blk = i_blk_pair - i_blk * NB
    if j_blk > i_blk:
        return

    row_start = i_t * BT + i_blk * BC
    col_start = j_blk * BC

    b_L = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        k_off = i_k * BK
        o_k = k_off + tl.arange(0, BK)
        m_k = o_k < K

        p_k_row = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (row_start, k_off),
            (BC, BK),
            (1, 0),
        )
        p_g_row = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (row_start, k_off),
            (BC, BK),
            (1, 0),
        )
        p_k_col = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + col_start, k_off),
            (BC, BK),
            (1, 0),
        )
        p_g_col = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + col_start, k_off),
            (BC, BK),
            (1, 0),
        )

        b_k_row = tl.load(p_k_row, boundary_check=(0, 1)).to(tl.float32)
        b_g_row = tl.load(p_g_row, boundary_check=(0, 1)).to(tl.float32)
        b_k_col = tl.load(p_k_col, boundary_check=(0, 1)).to(tl.float32)
        b_g_col = tl.load(p_g_col, boundary_check=(0, 1)).to(tl.float32)

        g_base = g + (bos * H + i_h) * K
        g_n = tl.load(g_base + row_start * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        row_scale = exp2(b_g_row - g_n[None, :])
        col_scale = exp2(g_n[None, :] - b_g_col)
        b_L += tl.dot(b_k_row * row_scale, tl.trans(b_k_col * col_scale))

    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (row_start,), (BC,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    b_L *= b_beta[:, None]

    if i_blk == j_blk:
        row_ids = row_start + tl.arange(0, BC)
        col_ids = i_t * BT + col_start + tl.arange(0, BC)
        token_start = row_ids - (row_ids % 2)
        m_keep = col_ids[None, :] < token_start[:, None]
        b_L = tl.where(m_keep, b_L, 0.0)

    p_L = tl.make_block_ptr(
        L + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (row_start, col_start),
        (BC, BC),
        (1, 0),
    )
    tl.store(p_L, b_L.to(p_L.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_kda_rank2_fwd_kernel_inter_solve_fused_bt64(
    k,
    g,
    beta,
    Akkd,
    Akk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    BT=64, BC=32 specialization of the fused inter + solve kernel.

    Computes the off-diagonal block L10 (rows [32:64], cols [0:32]) for a pseudo-time chunk
    and inverts the 2x2 block unit-lower-triangular system:

      [I+L00,    0]
      [  L10, I+L11]

    Diagonal block strictly-lower entries are read from `Akkd` (fp32).
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC

    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    beta += bos * H + i_h
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T

    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_g0 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        if i_tc1 < T:
            p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)

            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn1 = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            b_Akk10 += tl.dot(b_k1 * b_gqn1, b_kgt)

    if i_tc1 < T:
        p_b1 = tl.make_block_ptr(beta, (T,), (H,), (i_tc1,), (BC,), (0,))
        b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
        b_Akk10 *= b_b1[:, None]

    p_Akk00 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)

    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Ai00 = -tl.where(m_A, b_Ai00, 0)
    b_Ai11 = -tl.where(m_A, b_Ai11, 0)

    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
        b_a00 = tl.where(o_i < i, b_a00, 0.0)
        b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(BC + 2, min(2 * BC, T - i_tc0)):
        b_a11 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
        b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
        b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)

    b_Ai00 += m_I
    b_Ai11 += m_I

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BH": BH}, num_warps=num_warps)
        for BH in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["H"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "N"])
def chunk_kda_rank2_bwd_kernel_mask_dAkk_within_token(
    dAkk,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    In-place mask for rank-2 pseudo-time: zero dAkk[i, i-1] when floor(i/2)==floor((i-1)/2),
    i.e. for odd i within each sequence.

    dAkk is [B, T, H, BT] with BT=64, storing a BT-sized row window for each time step.
    """
    i_tg, i_hg = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = 0
        left, right = 0, N
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left

        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T

    if i_t >= T:
        return

    o_h = tl.arange(0, BH)
    m_h = (i_hg * BH + o_h) < H

    # Mask only odd pseudo-time indices; the masked column must be within the local BT window.
    if (i_t % 2) == 1:
        col = (i_t % BT) - 1
        # col is always valid for odd i_t since BT is even; still guard for clarity.
        if col >= 0:
            base = (bos + i_t) * H * BT + (i_hg * BH + o_h) * BT + col
            tl.store(dAkk + base, tl.zeros([BH], dtype=dAkk.dtype.element_ty), mask=m_h)


def chunk_kda_rank2_fwd_intra_a_inv(
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    *,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Build the pseudo-time (rank-interleaved) intra-chunk inverse matrix A^{-1} = (I + L)^{-1}.

    Shapes:
      k:    [B, T, H, K]    (pseudo-time, T = 2 * T_tokens)
      gk:   [B, T, H, K]    (chunk-local cumsum, log2)
      beta: [B, T, H]

    Returns:
      Akk:  [B, T, H, BT]   where BT=chunk_size (currently 64).
    """
    B, T, H, K = k.shape
    BT = int(chunk_size)
    BC = 32
    if BT not in (64, 128, 256):
        raise ValueError(
            f"chunk_kda_rank2_fwd_intra_a_inv currently supports chunk_size in {{64,128,256}}, got {chunk_size}.",
        )

    if cu_seqlens is not None:
        if k.shape[0] != 1:
            raise ValueError("When cu_seqlens is provided, expected batch size B==1 for packed inputs.")
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
        N = len(cu_seqlens) - 1
    else:
        NT = triton.cdiv(T, BT)
        N = B

    if BT in (64, 128):
        # Diagonal blocks buffer (fp32). Only strictly-lower entries are used.
        Akkd = torch.empty((B, T, H, BC), device=k.device, dtype=torch.float32)
        # Output inverse (dtype like k). Must be zero-initialized (kernel writes only lower blocks).
        Akk = torch.zeros((B, T, H, BT), device=k.device, dtype=k.dtype)

        def grid_diag(meta):
            return (B * T, triton.cdiv(H, meta["BH"]))

        chunk_kda_rank2_fwd_kernel_intra_token_parallel[grid_diag](
            k=k,
            g=gk,
            beta=beta,
            Akkd=Akkd,
            cu_seqlens=cu_seqlens,
            N=N,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
        )

        grid = (NT, B * H)
        if BT == 64:
            chunk_kda_rank2_fwd_kernel_inter_solve_fused_bt64[grid](
                k=k,
                g=gk,
                beta=beta,
                Akkd=Akkd,
                Akk=Akk,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=T,
                H=H,
                K=K,
                BT=BT,
                BC=BC,
            )
        else:
            chunk_kda_rank2_fwd_kernel_inter_solve_fused[grid](
                k=k,
                g=gk,
                beta=beta,
                Akkd=Akkd,
                Akk=Akk,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=T,
                H=H,
                K=K,
                BT=BT,
                BC=BC,
            )
        return Akk

    # BT=256: build L in float32 blocks then solve (I+L)^-1 via solve_tril (supports BT=256).
    NB = BT // BC
    if NB * BC != BT:
        raise ValueError(f"Expected chunk_size BT divisible by BC={BC}, got BT={BT}.")

    L = torch.empty((B, T, H, BT), device=k.device, dtype=torch.float32)
    grid_L = (NT, NB * NB, B * H)
    chunk_kda_rank2_fwd_kernel_build_L[grid_L](
        k=k,
        g=gk,
        beta=beta,
        L=L,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        NB=NB,
    )
    return solve_tril(A=L, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype)


def chunk_kda_rank2_bwd_mask_dAkk_within_token(
    dAkk: torch.Tensor,
    *,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    In-place mask for dAkk produced by `prepare_wy_repr_bwd` when using rank-2 pseudo-time.

    Zeroes the within-token coupling entry (i odd, j=i-1) for each sequence.
    """
    B, T, H, BT = dAkk.shape
    if int(chunk_size) != BT or BT not in (64, 128, 256):
        raise ValueError(
            "chunk_kda_rank2_bwd_mask_dAkk_within_token requires chunk_size == dAkk.shape[-1] and "
            f"BT in {{64,128,256}}, got {chunk_size=} {BT=}.",
        )
    if cu_seqlens is not None and B != 1:
        raise ValueError("When cu_seqlens is provided, expected batch size B==1 for packed inputs.")
    N = (len(cu_seqlens) - 1) if cu_seqlens is not None else B

    def grid(meta):
        return (B * T, triton.cdiv(H, meta["BH"]))

    chunk_kda_rank2_bwd_kernel_mask_dAkk_within_token[grid](
        dAkk=dAkk,
        cu_seqlens=cu_seqlens,
        N=N,
        T=T,
        H=H,
        BT=BT,
    )
    return dAkk
