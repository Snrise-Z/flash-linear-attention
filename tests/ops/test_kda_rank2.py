# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda import chunk_kda_rank2
from fla.ops.mkda.recurrent import mkda_recurrent
from fla.utils import IS_INTEL_ALCHEMIST, device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_kda_rank2_matches_mkda_recurrent(dtype: torch.dtype):
    if device == "cpu":
        pytest.skip(reason="Triton kernels are not available on CPU.")
    if IS_INTEL_ALCHEMIST:
        pytest.skip(reason="KDA kernels are not supported on alchemist in CI configs.")

    torch.manual_seed(0)
    B, T, H, K, V = 2, 33, 3, 64, 48
    scale = K**-0.5

    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, 2, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, 2, V, device=device, dtype=dtype)
    log_alpha = -F.softplus(torch.randn(B, T, H, K, device=device, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(B, T, H, 2, device=device, dtype=torch.float32)).to(dtype)
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32)

    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    tri, tri_ht = chunk_kda_rank2(
        q=q,
        k=k,
        v=v,
        log_alpha=log_alpha,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    ref, ref_ht = mkda_recurrent(
        q=q.to(torch.float32),
        k=k.to(torch.float32),
        v=v.to(torch.float32),
        log_alpha=log_alpha.to(torch.float32),
        beta=beta.to(torch.float32),
        initial_state=h0,
        scale=scale,
        output_final_state=True,
    )

    torch.testing.assert_close(tri.to(torch.float32), ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(tri_ht.to(torch.float32), ref_ht.to(torch.float32), rtol=2e-2, atol=2e-2)


def test_chunk_kda_rank2_backward_matches_reference():
    if device == "cpu":
        pytest.skip(reason="Triton kernels are not available on CPU.")
    if IS_INTEL_ALCHEMIST:
        pytest.skip(reason="KDA kernels are not supported on alchemist in CI configs.")

    torch.manual_seed(0)
    B, T, H, K, V = 2, 25, 2, 48, 32
    scale = K**-0.5

    q = F.normalize(torch.randn(B, T, H, K, device=device, dtype=torch.float16), p=2, dim=-1).requires_grad_(True)
    k = F.normalize(torch.randn(B, T, H, 2, K, device=device, dtype=torch.float16), p=2, dim=-1).requires_grad_(True)
    v = torch.randn(B, T, H, 2, V, device=device, dtype=torch.float16).requires_grad_(True)
    log_alpha = (-F.softplus(torch.randn(B, T, H, K, device=device, dtype=torch.float32))).to(torch.float16).requires_grad_(True)
    beta = torch.sigmoid(torch.randn(B, T, H, 2, device=device, dtype=torch.float32)).to(torch.float16).requires_grad_(True)
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32).requires_grad_(True)

    do = torch.randn(B, T, H, V, device=device, dtype=torch.float32)
    dht = torch.randn(B, H, K, V, device=device, dtype=torch.float32)

    tri, tri_ht = chunk_kda_rank2(
        q=q,
        k=k,
        v=v,
        log_alpha=log_alpha,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    ((tri.to(torch.float32) * do).sum() + (tri_ht.to(torch.float32) * dht).sum()).backward(retain_graph=True)
    tri_grads = (q.grad, k.grad, v.grad, log_alpha.grad, beta.grad, h0.grad)
    q.grad = k.grad = v.grad = log_alpha.grad = beta.grad = h0.grad = None

    ref, ref_ht = mkda_recurrent(
        q=q.to(torch.float32),
        k=k.to(torch.float32),
        v=v.to(torch.float32),
        log_alpha=log_alpha.to(torch.float32),
        beta=beta.to(torch.float32),
        initial_state=h0.to(torch.float32),
        scale=scale,
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_grads = (q.grad, k.grad, v.grad, log_alpha.grad, beta.grad, h0.grad)

    names = ("dq", "dk", "dv", "dlog_alpha", "dbeta", "dh0")
    for name, tri_g, ref_g in zip(names, tri_grads, ref_grads, strict=True):
        torch.testing.assert_close(tri_g.to(torch.float32), ref_g.to(torch.float32), rtol=2e-3, atol=2e-3, msg=name)
