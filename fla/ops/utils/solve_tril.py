# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.utils.op import make_tensor_descriptor
from fla.utils import IS_TMA_SUPPORTED, autotune_cache_kwargs, input_guard

FLA_TRIL_PRECISION = os.environ.get('FLA_TRIL_PRECISION', 'ieee')
assert FLA_TRIL_PRECISION in ['ieee', 'tf32', 'tf32x3'], \
    f"FLA_TRIL_PRECISION must be one of 'ieee', 'tf32', or 'tf32x3', but got {FLA_TRIL_PRECISION}"
DOT_PRECISION_AUTOTUNE_LIST = ["ieee"] if not IS_TMA_SUPPORTED else list({"ieee", FLA_TRIL_PRECISION})


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': 'ieee'}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def solve_tril_16x16_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
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
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A = A + (bos*H + i_h) * BT
    Ai = Ai + (bos*H + i_h) * 16

    offset = (i_t * 16) % BT
    if not USE_TMA:
        p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * 16, offset), (16, 16), (1, 0))
        # [16, 16]
        b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
        b_A = tl.where(m_A, b_A, 0)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, 16], [H*16, 1], [16, 16])
        b_A = desc.load([i_t * 16, offset]).to(tl.float32)
        b_A = tl.where(m_A, b_A, 0)
    b_A = -b_A

    for i in range(2, min(16, T - i_t * 16)):
        # [16]
        b_a = -tl.load(A + (i_t * 16 + i) * H*BT + o_i + offset)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I
    if not USE_TMA:
        p_Ai = tl.make_block_ptr(Ai, (T, 16), (H*16, 1), (i_t * 16, 0), (16, 16), (1, 0))
        tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        desc_o.store([i_t * 16, 0], b_A.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=['H', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
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

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    if not USE_TMA:
        p_A_11 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_A_22 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H*BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)

    b_Ai_11 += m_I
    b_Ai_22 += m_I

    if not USE_TMA:
        p_A_21 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)

    if not USE_TMA:
        p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        desc_o.store([i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4, 5]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=['H', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
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

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    if not USE_TMA:
        p_A_11 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_A_22 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        p_A_33 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
        p_A_44 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H*BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)
        b_Ai_33 = desc.load([i_t * BT + 32, 32]).to(tl.float32)
        b_Ai_44 = desc.load([i_t * BT + 48, 48]).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
        b_a_11 = tl.where(o_i < i, b_a_11, 0.)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
        b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 32)
        b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 48)
        b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    if not USE_TMA:
        p_A_21 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_A_31 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
        p_A_32 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
        p_A_41 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
        p_A_42 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
        p_A_43 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
        b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
        b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
        b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
        b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
        b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
        b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)
        b_A_31 = desc.load([i_t * BT + 32, 0]).to(tl.float32)
        b_A_32 = desc.load([i_t * BT + 32, 16]).to(tl.float32)
        b_A_41 = desc.load([i_t * BT + 48, 0]).to(tl.float32)
        b_A_42 = desc.load([i_t * BT + 48, 16]).to(tl.float32)
        b_A_43 = desc.load([i_t * BT + 48, 32]).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)
    b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION), b_Ai_22, input_precision=DOT_PRECISION)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION), b_Ai_33, input_precision=DOT_PRECISION)

    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) +
        tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION) +
        tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    if not USE_TMA:
        p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
        p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
        p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_Ai_31 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
        p_Ai_32 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
        p_Ai_41 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
        p_Ai_42 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
        p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
        tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_31, b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_32, b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_41, b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_42, b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_43, b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        desc_o.store([i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 32], b_Ai_33.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 48], b_Ai_44.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 0], b_Ai_31.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 16], b_Ai_32.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 0], b_Ai_41.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 16], b_Ai_42.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 32], b_Ai_43.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4, 5]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=['H', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_128x128_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
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

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    if USE_TMA:
        desc = make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H * BT, 1], [16, 16])

    # Diagonal 16x16 inverses for each 16-block along the diagonal.
    b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32) if USE_TMA else tl.load(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 0, 0), (16, 16), (1, 0)), boundary_check=(0, 1)
    ).to(tl.float32)
    b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32) if USE_TMA else tl.load(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)), boundary_check=(0, 1)
    ).to(tl.float32)
    b_Ai_33 = desc.load([i_t * BT + 32, 32]).to(tl.float32) if USE_TMA else tl.load(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)), boundary_check=(0, 1)
    ).to(tl.float32)
    b_Ai_44 = desc.load([i_t * BT + 48, 48]).to(tl.float32) if USE_TMA else tl.load(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)), boundary_check=(0, 1)
    ).to(tl.float32)
    b_Ai_55 = desc.load([i_t * BT + 64, 64]).to(tl.float32) if USE_TMA else tl.load(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 64, 64), (16, 16), (1, 0)), boundary_check=(0, 1)
    ).to(tl.float32)
    b_Ai_66 = desc.load([i_t * BT + 80, 80]).to(tl.float32) if USE_TMA else tl.load(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 80, 80), (16, 16), (1, 0)), boundary_check=(0, 1)
    ).to(tl.float32)
    b_Ai_77 = desc.load([i_t * BT + 96, 96]).to(tl.float32) if USE_TMA else tl.load(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 96, 96), (16, 16), (1, 0)), boundary_check=(0, 1)
    ).to(tl.float32)
    b_Ai_88 = desc.load([i_t * BT + 112, 112]).to(tl.float32) if USE_TMA else tl.load(
        tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + 112, 112), (16, 16), (1, 0)), boundary_check=(0, 1)
    ).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)
    b_Ai_55 = -tl.where(m_A, b_Ai_55, 0)
    b_Ai_66 = -tl.where(m_A, b_Ai_66, 0)
    b_Ai_77 = -tl.where(m_A, b_Ai_77, 0)
    b_Ai_88 = -tl.where(m_A, b_Ai_88, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a = tl.where(o_i < i - 16, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)
        b_a = tl.where(o_i < i - 32, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)
        b_a = tl.where(o_i < i - 48, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a, b_Ai_44)
    for i in range(64 + 2, min(80, T - i_t * BT)):
        b_a = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 64)
        b_a = tl.where(o_i < i - 64, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_55, 0)
        b_Ai_55 = tl.where((o_i == i - 64)[:, None], b_a, b_Ai_55)
    for i in range(80 + 2, min(96, T - i_t * BT)):
        b_a = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 80)
        b_a = tl.where(o_i < i - 80, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_66, 0)
        b_Ai_66 = tl.where((o_i == i - 80)[:, None], b_a, b_Ai_66)
    for i in range(96 + 2, min(112, T - i_t * BT)):
        b_a = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 96)
        b_a = tl.where(o_i < i - 96, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_77, 0)
        b_Ai_77 = tl.where((o_i == i - 96)[:, None], b_a, b_Ai_77)
    for i in range(112 + 2, min(128, T - i_t * BT)):
        b_a = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 112)
        b_a = tl.where(o_i < i - 112, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_88, 0)
        b_Ai_88 = tl.where((o_i == i - 112)[:, None], b_a, b_Ai_88)

    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I
    b_Ai_55 += m_I
    b_Ai_66 += m_I
    b_Ai_77 += m_I
    b_Ai_88 += m_I

    # Store diagonal blocks.
    if USE_TMA:
        desc_o.store([i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 32], b_Ai_33.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 48], b_Ai_44.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 64, 64], b_Ai_55.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 80, 80], b_Ai_66.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 96, 96], b_Ai_77.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 112, 112], b_Ai_88.to(desc_o.dtype, fp_downcast_rounding="rtne"))
    else:
        p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 0, 0), (16, 16), (1, 0))
        p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
        p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
        p_Ai_55 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 64, 64), (16, 16), (1, 0))
        p_Ai_66 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 80, 80), (16, 16), (1, 0))
        p_Ai_77 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 96, 96), (16, 16), (1, 0))
        p_Ai_88 = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + 112, 112), (16, 16), (1, 0))
        tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_33, b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_44, b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_55, b_Ai_55.to(p_Ai_55.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_66, b_Ai_66.to(p_Ai_66.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_77, b_Ai_77.to(p_Ai_77.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_88, b_Ai_88.to(p_Ai_88.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

    # Compute all off-diagonal 16x16 blocks of the inverse with block forward substitution.
    # Block indices are 0..7, mapping to rows/cols [0,16,32,...,112] within the chunk.
    diag = (b_Ai_11, b_Ai_22, b_Ai_33, b_Ai_44, b_Ai_55, b_Ai_66, b_Ai_77, b_Ai_88)
    for i_blk in range(1, 8):
        ai_ii = diag[i_blk]
        row_i = i_t * BT + i_blk * 16
        for j_blk in range(0, i_blk):
            acc = tl.zeros([16, 16], dtype=tl.float32)
            col_j = j_blk * 16
            for k_blk in range(j_blk, i_blk):
                if USE_TMA:
                    a_ik = desc.load([row_i, k_blk * 16]).to(tl.float32)
                    ai_kj = desc_o.load([i_t * BT + k_blk * 16, col_j]).to(tl.float32)
                else:
                    p_Aik = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (row_i, k_blk * 16), (16, 16), (1, 0))
                    p_Aikj = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (i_t * BT + k_blk * 16, col_j), (16, 16), (1, 0))
                    a_ik = tl.load(p_Aik, boundary_check=(0, 1)).to(tl.float32)
                    ai_kj = tl.load(p_Aikj, boundary_check=(0, 1)).to(tl.float32)
                acc += tl.dot(a_ik, ai_kj, input_precision=DOT_PRECISION)
            ai_ij = -tl.dot(ai_ii, acc, input_precision=DOT_PRECISION)
            if USE_TMA:
                desc_o.store([row_i, col_j], ai_ij.to(desc_o.dtype, fp_downcast_rounding="rtne"))
            else:
                p = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (row_i, col_j), (16, 16), (1, 0))
                tl.store(p, ai_ij.to(p.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4, 5]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=['H', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_256x256_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
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

    if i_t * BT >= T:
        return

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    if USE_TMA:
        desc = make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H * BT, 1], [16, 16])

    # 1) Solve diagonal 16x16 blocks independently.
    for blk in range(0, 16):
        row_off = i_t * BT + blk * 16
        col_off = blk * 16

        if USE_TMA:
            b_L = desc.load([row_off, col_off]).to(tl.float32)
        else:
            p_L = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (row_off, col_off), (16, 16), (1, 0))
            b_L = tl.load(p_L, boundary_check=(0, 1)).to(tl.float32)

        b_Ai = -tl.where(m_A, b_L, 0)
        # Forward substitution within the 16x16 block.
        for i in range(2, 16):
            b_a = -b_L[i, :]
            b_a = tl.where(o_i < i, b_a, 0.)
            b_a += tl.sum(b_a[:, None] * b_Ai, 0)
            b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
        b_Ai += m_I

        if USE_TMA:
            desc_o.store([row_off, col_off], b_Ai.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        else:
            p_Ai = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (row_off, col_off), (16, 16), (1, 0))
            tl.store(p_Ai, b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

    # 2) Compute off-diagonal blocks with block forward substitution:
    #    Ai_{i,j} = -Ai_{i,i} @ sum_{k=j..i-1} L_{i,k} @ Ai_{k,j}.
    for i_blk in range(1, 16):
        row_i = i_t * BT + i_blk * 16

        if USE_TMA:
            b_Ai_ii = desc_o.load([row_i, i_blk * 16]).to(tl.float32)
        else:
            p_Ai_ii = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (row_i, i_blk * 16), (16, 16), (1, 0))
            b_Ai_ii = tl.load(p_Ai_ii, boundary_check=(0, 1)).to(tl.float32)

        for j_blk in range(0, i_blk):
            col_j = j_blk * 16
            acc = tl.zeros([16, 16], dtype=tl.float32)
            for k_blk in range(j_blk, i_blk):
                col_k = k_blk * 16
                row_k = i_t * BT + k_blk * 16

                if USE_TMA:
                    b_Lik = desc.load([row_i, col_k]).to(tl.float32)
                    b_Aikj = desc_o.load([row_k, col_j]).to(tl.float32)
                else:
                    p_Lik = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (row_i, col_k), (16, 16), (1, 0))
                    p_Aikj = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (row_k, col_j), (16, 16), (1, 0))
                    b_Lik = tl.load(p_Lik, boundary_check=(0, 1)).to(tl.float32)
                    b_Aikj = tl.load(p_Aikj, boundary_check=(0, 1)).to(tl.float32)

                acc += tl.dot(b_Lik, b_Aikj, input_precision=DOT_PRECISION)

            b_Ai_ij = -tl.dot(b_Ai_ii, acc, input_precision=DOT_PRECISION)

            if USE_TMA:
                desc_o.store([row_i, col_j], b_Ai_ij.to(desc_o.dtype, fp_downcast_rounding="rtne"))
            else:
                p_Ai_ij = tl.make_block_ptr(Ai, (T, BT), (H * BT, 1), (row_i, col_j), (16, 16), (1, 0))
                tl.store(p_Ai_ij, b_Ai_ij.to(p_Ai_ij.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, BT], where BT should only be 16, 32, 64, 128, or 256.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`.
            If `None`, the output dtype will be the same as the input dtype.

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64, 128, 256]
    output_dtype = A.dtype if output_dtype is None else output_dtype

    B, T, H, BT = A.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    Ai = torch.zeros_like(A, dtype=output_dtype)
    if BT == 16:
        merge_fn = solve_tril_16x16_kernel
    elif BT == 32:
        merge_fn = merge_16x16_to_32x32_inverse_kernel
    elif BT == 64:
        merge_fn = merge_16x16_to_64x64_inverse_kernel
    elif BT == 128:
        merge_fn = merge_16x16_to_128x128_inverse_kernel
    elif BT == 256:
        merge_fn = merge_16x16_to_256x256_inverse_kernel

    merge_fn[NT, B * H](
        A=A,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        USE_TMA=IS_TMA_SUPPORTED,
    )
    return Ai
