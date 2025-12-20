# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import pytest
import torch
import torch.nn.functional as F

from fla.ops.mkda import mkda_chunkwise_parallel, mkda_recurrent


@pytest.mark.parametrize("chunk_size", [4, 8])
def test_mkda_chunkwise_matches_recurrent_small(chunk_size: int):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, T, H, K, V, R = 2, 13, 3, 16, 12, 4
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    k = torch.randn(B, T, H, R, K, device=device, dtype=torch.float32)
    v = torch.randn(B, T, H, R, V, device=device, dtype=torch.float32)
    log_alpha = -F.softplus(torch.randn(B, T, H, K, device=device, dtype=torch.float32))  # <= 0
    beta = torch.sigmoid(torch.randn(B, T, H, R, device=device, dtype=torch.float32))

    # Match the layer behavior: normalize q/k and use float32 state.
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    ref, ref_s = mkda_recurrent(q, k, v, log_alpha, beta, output_final_state=True)
    tri, tri_s = mkda_chunkwise_parallel(q, k, v, log_alpha, beta, chunk_size=chunk_size, output_final_state=True)

    torch.testing.assert_close(tri, ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(tri_s, ref_s, rtol=2e-3, atol=2e-3)

