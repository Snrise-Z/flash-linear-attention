# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import IS_TF32_SUPPORTED, autotune_cache_kwargs, check_shared_mem


@triton.heuristics({
    'STORE_QG': lambda args: args['qg'] is not None,
    'STORE_KG': lambda args: args['kg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
        for DOT_PRECISION in (["tf32x3", "ieee"] if IS_TF32_SUPPORTED else ["ieee"])
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_b = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(w + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kb *= exp2(b_gk)
        if STORE_QG:
            p_q = tl.make_block_ptr(q + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_qg = tl.make_block_ptr(qg + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.)
            b_kg = b_k * tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], exp2(b_gn[None, :] - b_gk), 0)
            p_kg = tl.make_block_ptr(kg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
        for DOT_PRECISION in (["tf32x3", "ieee"] if IS_TF32_SUPPORTED else ["ieee"])
    ],
    key=['H', 'V', 'BT', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_u_fwd_kernel_tiled(
    v,
    beta,
    u,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    BM: tl.constexpr,
    BKA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """
    Tiled u = A @ (v * beta) for BT>128 (e.g. BT=256).

    Grid: (NT, B*H, NM*NV), where NM=BT/BM and NV=ceil(V/BV).
    """
    i_t, i_bh, i_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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

    NV: tl.constexpr = tl.cdiv(V, BV)
    i_m = i_tile // NV
    i_v = i_tile - i_m * NV

    row_start = i_t * BT + i_m * BM
    col_v = i_v * BV

    base_v = v + (bos * H + i_h) * V
    base_u = u + (bos * H + i_h) * V
    base_b = beta + bos * H + i_h
    base_A = A + (bos * H + i_h) * BT

    acc = tl.zeros([BM, BV], dtype=tl.float32)
    for k_inner in range(0, BT, BKA):
        p_A = tl.make_block_ptr(base_A, (T, BT), (H * BT, 1), (row_start, k_inner), (BM, BKA), (1, 0))
        b_A = tl.load(p_A, boundary_check=(0, 1))

        p_v = tl.make_block_ptr(base_v, (T, V), (H * V, 1), (i_t * BT + k_inner, col_v), (BKA, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        p_b = tl.make_block_ptr(base_b, (T,), (H,), (i_t * BT + k_inner,), (BKA,), (0,))
        b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)

        acc += tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)

    p_u = tl.make_block_ptr(base_u, (T, V), (H * V, 1), (row_start, col_v), (BM, BV), (1, 0))
    tl.store(p_u, acc.to(p_u.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'STORE_QG': lambda args: args['qg'] is not None,
    'STORE_KG': lambda args: args['kg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
        for DOT_PRECISION in (["tf32x3", "ieee"] if IS_TF32_SUPPORTED else ["ieee"])
    ],
    key=['H', 'K', 'BT', 'BK', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_fwd_kernel_tiled(
    q,
    k,
    qg,
    kg,
    beta,
    w,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    BKA: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """
    Tiled w = A @ (k * beta * exp2(gk)) for BT>128 (e.g. BT=256), and optionally:
      qg = q * exp2(gk)
      kg = k * exp2(g_last - gk)

    Grid: (NT, B*H, NM*NK), where NM=BT/BM and NK=ceil(K/BK).
    """
    i_t, i_bh, i_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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

    NK: tl.constexpr = tl.cdiv(K, BK)
    i_m = i_tile // NK
    i_k = i_tile - i_m * NK

    row_start = i_t * BT + i_m * BM
    col_k = i_k * BK

    base_k = k + (bos * H + i_h) * K
    base_w = w + (bos * H + i_h) * K
    base_b = beta + bos * H + i_h
    base_A = A + (bos * H + i_h) * BT
    base_gk = gk + (bos * H + i_h) * K

    p_b_row = tl.make_block_ptr(base_b, (T,), (H,), (row_start,), (BM,), (0,))
    b_b_row = tl.load(p_b_row, boundary_check=(0,)).to(tl.float32)

    p_k_row = tl.make_block_ptr(base_k, (T, K), (H * K, 1), (row_start, col_k), (BM, BK), (1, 0))
    b_k_row = tl.load(p_k_row, boundary_check=(0, 1))

    p_gk_row = tl.make_block_ptr(base_gk, (T, K), (H * K, 1), (row_start, col_k), (BM, BK), (1, 0))
    b_gk_row = tl.load(p_gk_row, boundary_check=(0, 1))

    if STORE_QG:
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (row_start, col_k), (BM, BK), (1, 0))
        p_qg = tl.make_block_ptr(qg + (bos * H + i_h) * K, (T, K), (H * K, 1), (row_start, col_k), (BM, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_qg = b_q * exp2(b_gk_row)
        tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))

    if STORE_KG:
        last_idx = min(i_t * BT + BT, T) - 1
        o_k = col_k + tl.arange(0, BK)
        m_k = o_k < K
        b_gn = tl.load(base_gk + last_idx * H * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        o_row = row_start + tl.arange(0, BM)
        m_row = o_row < T
        b_kg = b_k_row * tl.where(m_row[:, None], exp2(b_gn[None, :] - b_gk_row), 0.0)
        p_kg = tl.make_block_ptr(kg + (bos * H + i_h) * K, (T, K), (H * K, 1), (row_start, col_k), (BM, BK), (1, 0))
        tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

    acc = tl.zeros([BM, BK], dtype=tl.float32)
    for k_inner in range(0, BT, BKA):
        p_A = tl.make_block_ptr(base_A, (T, BT), (H * BT, 1), (row_start, k_inner), (BM, BKA), (1, 0))
        b_A = tl.load(p_A, boundary_check=(0, 1))

        p_k_rhs = tl.make_block_ptr(base_k, (T, K), (H * K, 1), (i_t * BT + k_inner, col_k), (BKA, BK), (1, 0))
        b_k_rhs = tl.load(p_k_rhs, boundary_check=(0, 1))
        p_gk_rhs = tl.make_block_ptr(base_gk, (T, K), (H * K, 1), (i_t * BT + k_inner, col_k), (BKA, BK), (1, 0))
        b_gk_rhs = tl.load(p_gk_rhs, boundary_check=(0, 1))
        p_b_rhs = tl.make_block_ptr(base_b, (T,), (H,), (i_t * BT + k_inner,), (BKA,), (0,))
        b_b_rhs = tl.load(p_b_rhs, boundary_check=(0,)).to(tl.float32)

        b_rhs = (b_k_rhs * b_b_rhs[:, None] * exp2(b_gk_rhs)).to(b_k_rhs.dtype)
        acc += tl.dot(b_A, b_rhs, input_precision=DOT_PRECISION)

    p_w = tl.make_block_ptr(base_w, (T, K), (H * K, 1), (row_start, col_k), (BM, BK), (1, 0))
    tl.store(p_w, acc.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    beta,
    gk,
    A,
    dA,
    dw,
    du,
    dk,
    dk2,
    dv,
    db,
    dg,
    dg2,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_b = tl.make_block_ptr(beta + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_db = tl.make_block_ptr(db + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_b = tl.load(p_b, boundary_check=(0,))
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk2 = tl.make_block_ptr(dk2 + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg2 = tl.make_block_ptr(dg2 + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        # [BT, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_gk_exp = exp2(tl.load(p_gk, boundary_check=(0, 1)))
        b_kbg = b_k * b_b[:, None] * b_gk_exp
        b_dw = tl.load(p_dw, boundary_check=(0, 1))

        b_dA += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype))
        b_dkbg = tl.dot(b_A, b_dw)
        b_dk = b_dkbg * b_gk_exp * b_b[:, None] + tl.load(p_dk, boundary_check=(0, 1))
        b_db += tl.sum(b_dkbg * b_k * b_gk_exp, 1)
        b_dg = b_kbg * b_dkbg + tl.load(p_dg, boundary_check=(0, 1))

        tl.store(p_dk2, b_dk.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg2, b_dg.to(p_dg2.dtype.element_ty), boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA += tl.dot(b_du, tl.trans(b_vb))
        b_dvb = tl.dot(b_A, b_du)
        b_dv = b_dvb * b_b[:, None]
        b_db += tl.sum(b_dvb * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))

    b_dA = tl.where(m_A, -b_dA, 0)

    # if using gk, save dA first and handle dk in another kernel
    p_dA = tl.make_block_ptr(dA + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'BK', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel_tiled_k(
    k,
    beta,
    gk,
    w,
    A,
    dw,
    dk,
    dk2,
    db,
    dg,
    dg2,
    dA,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    BKA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    K-path tiled backward for BT>128:
      d_kbg = A^T @ dw
      dk   += d_kbg * exp2(gk) * beta
      dg   += (k * beta * exp2(gk)) * d_kbg
      db   += sum_k d_kbg * k * exp2(gk)
      dL   += -(d_kbg @ w^T)  (strict lower only; atomic add)

    Grid: (NT, B*H, NM*NK), where NM=BT/BM and NK=ceil(K/BK).
    """
    i_t, i_bh, i_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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

    NK: tl.constexpr = tl.cdiv(K, BK)
    i_m = i_tile // NK
    i_k = i_tile - i_m * NK

    row_start = i_t * BT + i_m * BM
    row_off = i_m * BM
    col_k = i_k * BK

    base_k = k + (bos * H + i_h) * K
    base_w = w + (bos * H + i_h) * K
    base_b = beta + bos * H + i_h
    base_A = A + (bos * H + i_h) * BT
    base_dw = dw + (bos * H + i_h) * K
    base_dk = dk + (bos * H + i_h) * K
    base_dk2 = dk2 + (bos * H + i_h) * K
    base_dg = dg + (bos * H + i_h) * K
    base_dg2 = dg2 + (bos * H + i_h) * K
    base_gk = gk + (bos * H + i_h) * K

    # d_kbg_tile = A^T @ dw
    d_kbg = tl.zeros([BM, BK], dtype=tl.float32)
    for t_inner in range(0, BT, BKA):
        # A^T view: shape (BT, T), strides (1, H*BT)
        p_AT = tl.make_block_ptr(base_A, (BT, T), (1, H * BT), (row_off, i_t * BT + t_inner), (BM, BKA), (0, 1))
        b_AT = tl.load(p_AT, boundary_check=(0, 1))

        p_dw = tl.make_block_ptr(base_dw, (T, K), (H * K, 1), (i_t * BT + t_inner, col_k), (BKA, BK), (1, 0))
        b_dw = tl.load(p_dw, boundary_check=(0, 1))

        d_kbg += tl.dot(b_AT, b_dw)

    # Load per-row beta/gk/k for this output tile.
    p_b = tl.make_block_ptr(base_b, (T,), (H,), (row_start,), (BM,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)

    p_k = tl.make_block_ptr(base_k, (T, K), (H * K, 1), (row_start, col_k), (BM, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    p_gk = tl.make_block_ptr(base_gk, (T, K), (H * K, 1), (row_start, col_k), (BM, BK), (1, 0))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_gk_exp = exp2(b_gk)

    # dk / dg updates (no atomics; each (row,col) owned by exactly one program).
    p_dk_in = tl.make_block_ptr(base_dk, (T, K), (H * K, 1), (row_start, col_k), (BM, BK), (1, 0))
    p_dk_out = tl.make_block_ptr(base_dk2, (T, K), (H * K, 1), (row_start, col_k), (BM, BK), (1, 0))
    dk_val = d_kbg * b_gk_exp * b_b[:, None] + tl.load(p_dk_in, boundary_check=(0, 1))
    tl.store(p_dk_out, dk_val.to(p_dk_out.dtype.element_ty), boundary_check=(0, 1))

    p_dg_in = tl.make_block_ptr(base_dg, (T, K), (H * K, 1), (row_start, col_k), (BM, BK), (1, 0))
    p_dg_out = tl.make_block_ptr(base_dg2, (T, K), (H * K, 1), (row_start, col_k), (BM, BK), (1, 0))
    kbg = b_k * b_b[:, None] * b_gk_exp
    dg_val = kbg * d_kbg + tl.load(p_dg_in, boundary_check=(0, 1))
    tl.store(p_dg_out, dg_val.to(p_dg_out.dtype.element_ty), boundary_check=(0, 1))

    # Atomic accumulate db over K tiles: db += sum_k d_kbg * k * exp2(gk).
    db_part = tl.sum(d_kbg * b_k.to(tl.float32) * b_gk_exp, axis=1)
    base_db = db + (bos * H + i_h)
    row_ids = row_start + tl.arange(0, BM)
    m_row = row_ids < T
    tl.atomic_add(base_db + row_ids * H, db_part, mask=m_row)

    # Atomic accumulate dA blocks over K tiles: dA += -(d_kbg @ w^T) for strict-lower entries.
    NB: tl.constexpr = BT // BM
    base_dA = dA + (bos * H + i_h) * BT
    row_ptr_offs = row_ids * H * BT
    for j_blk in range(0, NB):
        if j_blk <= i_m:
            col_start = j_blk * BM
            p_w_col = tl.make_block_ptr(
                base_w, (T, K), (H * K, 1), (i_t * BT + col_start, col_k), (BM, BK), (1, 0)
            )
            b_w_col = tl.load(p_w_col, boundary_check=(0, 1))
            part = tl.dot(d_kbg.to(b_w_col.dtype), tl.trans(b_w_col))

            col_ids = col_start + tl.arange(0, BM)
            ptrs = base_dA + row_ptr_offs[:, None] + col_ids[None, :]

            if j_blk == i_m:
                m_lower = tl.arange(0, BM)[:, None] > tl.arange(0, BM)[None, :]
                tl.atomic_add(ptrs, -part, mask=m_row[:, None] & m_lower)
            else:
                tl.atomic_add(ptrs, -part, mask=m_row[:, None])


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'V', 'BT', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel_tiled_v(
    v,
    beta,
    u,
    A,
    du,
    dv,
    db,
    dA,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    BM: tl.constexpr,
    BKA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    V-path tiled backward for BT>128:
      d_vb = A^T @ du
      dv   = d_vb * beta
      db  += sum_v d_vb * v
      dL  += -(d_vb @ u^T) (strict lower only; atomic add)

    Grid: (NT, B*H, NM*NV), where NM=BT/BM and NV=ceil(V/BV).
    """
    i_t, i_bh, i_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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

    NV: tl.constexpr = tl.cdiv(V, BV)
    i_m = i_tile // NV
    i_v = i_tile - i_m * NV

    row_start = i_t * BT + i_m * BM
    row_off = i_m * BM
    col_v = i_v * BV

    base_v = v + (bos * H + i_h) * V
    base_u = u + (bos * H + i_h) * V
    base_b = beta + bos * H + i_h
    base_A = A + (bos * H + i_h) * BT
    base_du = du + (bos * H + i_h) * V
    base_dv = dv + (bos * H + i_h) * V

    d_vb = tl.zeros([BM, BV], dtype=tl.float32)
    for t_inner in range(0, BT, BKA):
        p_AT = tl.make_block_ptr(base_A, (BT, T), (1, H * BT), (row_off, i_t * BT + t_inner), (BM, BKA), (0, 1))
        b_AT = tl.load(p_AT, boundary_check=(0, 1))

        p_du = tl.make_block_ptr(base_du, (T, V), (H * V, 1), (i_t * BT + t_inner, col_v), (BKA, BV), (1, 0))
        b_du = tl.load(p_du, boundary_check=(0, 1))
        d_vb += tl.dot(b_AT, b_du)

    p_b = tl.make_block_ptr(base_b, (T,), (H,), (row_start,), (BM,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)

    dv_val = d_vb * b_b[:, None]
    p_dv = tl.make_block_ptr(base_dv, (T, V), (H * V, 1), (row_start, col_v), (BM, BV), (1, 0))
    tl.store(p_dv, dv_val.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    # Atomic accumulate db over V tiles: db += sum_v d_vb * v.
    p_v = tl.make_block_ptr(base_v, (T, V), (H * V, 1), (row_start, col_v), (BM, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    db_part = tl.sum(d_vb * b_v, axis=1)
    base_db = db + (bos * H + i_h)
    row_ids = row_start + tl.arange(0, BM)
    m_row = row_ids < T
    tl.atomic_add(base_db + row_ids * H, db_part, mask=m_row)

    # Atomic accumulate dA blocks over V tiles: dA += -(d_vb @ u^T) for strict-lower entries.
    NB: tl.constexpr = BT // BM
    base_dA = dA + (bos * H + i_h) * BT
    row_ptr_offs = row_ids * H * BT
    for j_blk in range(0, NB):
        if j_blk <= i_m:
            col_start = j_blk * BM
            p_u_col = tl.make_block_ptr(
                base_u, (T, V), (H * V, 1), (i_t * BT + col_start, col_v), (BM, BV), (1, 0)
            )
            b_u_col = tl.load(p_u_col, boundary_check=(0, 1))
            part = tl.dot(d_vb.to(b_u_col.dtype), tl.trans(b_u_col))

            col_ids = col_start + tl.arange(0, BM)
            ptrs = base_dA + row_ptr_offs[:, None] + col_ids[None, :]
            if j_blk == i_m:
                m_lower = tl.arange(0, BM)[:, None] > tl.arange(0, BM)[None, :]
                tl.atomic_add(ptrs, -part, mask=m_row[:, None] & m_lower)
            else:
                tl.atomic_add(ptrs, -part, mask=m_row[:, None])


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    qg = torch.empty_like(q) if q is not None else None
    kg = torch.empty_like(k) if gk is not None else None
    if BT <= 128:
        recompute_w_u_fwd_kernel[(NT, B * H)](
            q=q,
            k=k,
            qg=qg,
            kg=kg,
            v=v,
            beta=beta,
            w=w,
            u=u,
            A=A,
            gk=gk,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
        )
        return w, u, qg, kg

    # BT>128: use tiled matmuls to avoid materializing BT×BT in registers.
    if gk is None:
        raise ValueError("recompute_w_u_fwd requires `gk` when BT>128.")

    BM = 32
    BKA = 32
    if BT % BM != 0 or BT % BKA != 0:
        raise ValueError(f"Expected BT divisible by {BM} and {BKA}, got BT={BT}.")

    grid_u = (NT, B * H, (BT // BM) * triton.cdiv(V, BV))
    recompute_u_fwd_kernel_tiled[grid_u](
        v=v,
        beta=beta,
        u=u,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
        BM=BM,
        BKA=BKA,
    )

    grid_w = (NT, B * H, (BT // BM) * triton.cdiv(K, BK))
    recompute_w_fwd_kernel_tiled[grid_w](
        q=q,
        k=k,
        qg=qg,
        kg=kg,
        beta=beta,
        w=w,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
        BM=BM,
        BKA=BKA,
    )
    return w, u, qg, kg


def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
    A: torch.Tensor,
    dk: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    dg: torch.Tensor,
    *,
    w: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = int(A.shape[-1])
    if A.shape != (B, T, H, BT):
        raise ValueError(f"Expected A to be [B,T,H,BT], got shape={tuple(A.shape)} for B={B} T={T} H={H}.")
    if BT > 128 and (w is None or u is None):
        raise ValueError("prepare_wy_repr_bwd requires `w` and `u` when BT>128.")
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    dk2 = torch.empty_like(dk, dtype=torch.float)
    dv = torch.empty_like(v)
    dg2 = torch.empty_like(gk, dtype=torch.float)
    if BT <= 128:
        dA = torch.empty_like(A, dtype=torch.float)
        db = torch.empty_like(beta, dtype=torch.float)
        prepare_wy_repr_bwd_kernel[(NT, B * H)](
            k=k,
            v=v,
            beta=beta,
            gk=gk,
            A=A,
            dA=dA,
            dw=dw,
            du=du,
            dk=dk,
            dk2=dk2,
            dv=dv,
            db=db,
            dg=dg,
            dg2=dg2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
        )
        dk = dk2
        dg = dg2
        return dk, dv, db, dg, dA

    # BT>128: avoid BT×BT materialization. Use two tiled kernels (K-path and V-path) that
    # atomically accumulate into (db, dA) and write (dk2, dg2, dv) directly.
    BM = 32
    BKA = 32
    if BT % BM != 0 or BT % BKA != 0:
        raise ValueError(f"Expected BT divisible by {BM} and {BKA}, got BT={BT}.")
    if w is None or u is None:
        raise ValueError("Internal error: expected w/u for BT>128.")
    if w.shape != (B, T, H, K):
        raise ValueError(f"Expected w to be [B,T,H,K], got shape={tuple(w.shape)}.")
    if u.shape != (B, T, H, V):
        raise ValueError(f"Expected u to be [B,T,H,V], got shape={tuple(u.shape)}.")

    dA = torch.zeros_like(A, dtype=torch.float)
    db = torch.zeros_like(beta, dtype=torch.float)

    grid_k = (NT, B * H, (BT // BM) * triton.cdiv(K, BK))
    prepare_wy_repr_bwd_kernel_tiled_k[grid_k](
        k=k,
        beta=beta,
        gk=gk,
        w=w,
        A=A,
        dw=dw,
        dk=dk,
        dk2=dk2,
        db=db,
        dg=dg,
        dg2=dg2,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
        BM=BM,
        BKA=BKA,
    )

    grid_v = (NT, B * H, (BT // BM) * triton.cdiv(V, BV))
    prepare_wy_repr_bwd_kernel_tiled_v[grid_v](
        v=v,
        beta=beta,
        u=u,
        A=A,
        du=du,
        dv=dv,
        db=db,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
        BM=BM,
        BKA=BKA,
    )
    dk = dk2
    dg = dg2
    return dk, dv, db, dg, dA
