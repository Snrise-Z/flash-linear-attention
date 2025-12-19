# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda import (
    chunk_kda_rank_r_microstep,
    fused_recurrent_kda_rank_r_microstep,
)
from fla.ops.kda.gate import naive_kda_gate
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device


def _expand_microsteps_reference(
    *,
    q: torch.Tensor,  # [B,T,H,K]
    k: torch.Tensor,  # [B,T,H,R,K]
    v: torch.Tensor,  # [B,T,H,R,V]
    g: torch.Tensor,  # [B,T,H,K] (raw if use_gate_in_kernel else log-decay)
    beta: torch.Tensor,  # [B,T,H,R]
    use_gate_in_kernel: bool,
    fill_g_raw: float,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    B, T, H, K = q.shape
    R = k.shape[-2]
    V = v.shape[-1]

    q_micro = q.repeat_interleave(R, dim=1)  # [B, T*R, H, K]
    k_micro = k.permute(0, 1, 3, 2, 4).reshape(B, T * R, H, K)
    v_micro = v.permute(0, 1, 3, 2, 4).reshape(B, T * R, H, V)
    beta_micro = beta.permute(0, 1, 3, 2).reshape(B, T * R, H)

    g_rep = g.repeat_interleave(R, dim=1)
    if R == 1:
        g_micro_in = g_rep
    else:
        micro_rank = torch.arange(T * R, device=g.device) % R
        is_first = micro_rank == 0
        fill = (torch.tensor(fill_g_raw, dtype=g.dtype, device=g.device) if use_gate_in_kernel else g.new_tensor(0.0))
        g_micro_in = torch.where(is_first.view(1, -1, 1, 1), g_rep, fill)

    if use_gate_in_kernel:
        assert A_log is not None
        g_micro = naive_kda_gate(g_micro_in, A_log, dt_bias)
    else:
        g_micro = g_micro_in

    return q_micro, k_micro, v_micro, g_micro, beta_micro, R


@pytest.mark.parametrize("use_gate_in_kernel", [False, True])
def test_chunk_kda_rank_r_microstep_matches_naive(use_gate_in_kernel: bool):
    torch.manual_seed(42)
    if IS_INTEL_ALCHEMIST:
        pytest.skip(reason="KDA kernels are not supported on alchemist in CI configs.")

    B, T, H, D, R = 2, 33, 3, 64, 3
    V = D
    scale = 0.5
    fill_g_raw = -1.0e4

    q = torch.randn(B, T, H, D, dtype=torch.float16, device=device)
    k = torch.randn(B, T, H, R, D, dtype=torch.float16, device=device)
    v = torch.randn(B, T, H, R, V, dtype=torch.float16, device=device)
    beta = torch.randn(B, T, H, R, dtype=torch.float16, device=device).sigmoid()
    h0 = torch.randn(B, H, D, V, dtype=torch.float32, device=device)

    # Use external normalization to match other KDA tests.
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        g = torch.randn(B, T, H, D, dtype=torch.float16, device=device)
        A_log = torch.randn(H, dtype=torch.float32, device=device)
        dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)
    else:
        g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float32, device=device))

    q_micro, k_micro, v_micro, g_micro, beta_micro, R_out = _expand_microsteps_reference(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        use_gate_in_kernel=use_gate_in_kernel,
        fill_g_raw=fill_g_raw,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    assert R_out == R

    ref_micro, ref_ht = naive_recurrent_kda(
        q=q_micro,
        k=k_micro,
        v=v_micro,
        g=g_micro,
        beta=beta_micro,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    ref = ref_micro.reshape(B, T, R, H, V)[:, :, -1]

    tri, tri_ht = chunk_kda_rank_r_microstep(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=use_gate_in_kernel,
        fill_g_raw=fill_g_raw,
        A_log=A_log,
        dt_bias=dt_bias,
    )

    assert_close("o", ref, tri, 0.01)
    assert_close("ht", ref_ht, tri_ht, 0.01)


def test_fused_recurrent_kda_rank_r_microstep_matches_naive():
    torch.manual_seed(42)
    if IS_INTEL_ALCHEMIST:
        pytest.skip(reason="KDA kernels are not supported on alchemist in CI configs.")

    B, T, H, D, R = 2, 25, 2, 64, 4
    V = D
    scale = 1.0

    q = torch.randn(B, T, H, D, dtype=torch.float16, device=device)
    k = torch.randn(B, T, H, R, D, dtype=torch.float16, device=device)
    v = torch.randn(B, T, H, R, V, dtype=torch.float16, device=device)
    beta = torch.randn(B, T, H, R, dtype=torch.float16, device=device).sigmoid()
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float32, device=device))
    h0 = torch.randn(B, H, D, V, dtype=torch.float32, device=device)

    # Use external normalization to match other KDA tests.
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    q_micro, k_micro, v_micro, g_micro, beta_micro, _ = _expand_microsteps_reference(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        use_gate_in_kernel=False,
        fill_g_raw=-1.0e4,
        A_log=None,
        dt_bias=None,
    )
    ref_micro, ref_ht = naive_recurrent_kda(
        q=q_micro,
        k=k_micro,
        v=v_micro,
        g=g_micro,
        beta=beta_micro,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    ref = ref_micro.reshape(B, T, R, H, V)[:, :, -1]

    tri, tri_ht = fused_recurrent_kda_rank_r_microstep(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )

    assert_close("o", ref, tri, 0.01)
    assert_close("ht", ref_ht, tri_ht, 0.01)

