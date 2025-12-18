# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class FastSlowKimiDeltaAttention(nn.Module):
    """
    Fast/Slow Kimi Delta Attention (FKDA).

    This variant maintains two KDA memories (fast/slow) while keeping the recurrence affine in S,
    so chunk-parallel training remains supported.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        use_qk_l2norm_in_kernel: bool = True,
        # ablations
        fix_lambda: float | None = None,
        share_decay_gate: bool = False,
        **kwargs,
    ) -> FastSlowKimiDeltaAttention:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self.fix_lambda = fix_lambda
        self.share_decay_gate = share_decay_gate

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.")
        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )
        if mode not in ["chunk", "fused_recurrent"]:
            raise ValueError(f"Not supported mode `{mode}`.")
        if fix_lambda is not None and not (0.0 <= float(fix_lambda) <= 1.0):
            raise ValueError("fix_lambda must be in [0, 1] when set.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        # g_raw: base ± delta
        self.g_base_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.g_delta_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )

        # beta: base ± delta (per head)
        self.beta_base_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.beta_delta_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # lambda: per head fusion weight
        self.lambda_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True

        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None and attention_mask.ndim != 2:
            mask = attention_mask
            while mask.ndim > 2 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            if mask.ndim == 2:
                attention_mask = mask
            elif mask.ndim == 3:
                if mask.dtype == torch.bool:
                    keep = mask
                else:
                    keep = (mask == 0) if mask.min().item() < 0 else (mask > 0)
                attention_mask = keep.any(dim=-2).to(torch.bool)
            else:
                raise ValueError(
                    "Expected attention_mask to be 2D padding mask [batch, seq_len]. "
                    f"Got shape={tuple(attention_mask.shape)}.",
                )

        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training and mode != "chunk":
            raise AssertionError("Only chunk mode is supported in training.")

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        recurrent_fast, recurrent_slow = None, None
        if last_state is not None:
            rs = last_state["recurrent_state"]
            if isinstance(rs, (tuple, list)) and len(rs) == 2:
                recurrent_fast, recurrent_slow = rs
            else:
                recurrent_fast = rs
                recurrent_slow = None

        cu_seqlens = kwargs.get("cu_seqlens")
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        g_base = self.g_base_proj(hidden_states)
        g_delta = self.g_delta_proj(hidden_states)
        beta_base = self.beta_base_proj(hidden_states)
        beta_delta = self.beta_delta_proj(hidden_states)

        if self.fix_lambda is None:
            lam = self.lambda_proj(hidden_states).sigmoid()
        else:
            lam = torch.full_like(beta_base, float(self.fix_lambda))

        g_fast = g_base + g_delta
        if self.share_decay_gate:
            g_slow = g_fast
        else:
            g_slow = g_base - g_delta

        beta_base = beta_base.to(torch.float32)
        beta_delta = beta_delta.to(torch.float32)
        beta_fast = torch.sigmoid(beta_base + beta_delta).to(v.dtype)
        beta_slow = torch.sigmoid(beta_base - beta_delta).to(v.dtype)
        lam = lam.to(v.dtype)

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        g_fast = rearrange(g_fast, "... (h d) -> ... h d", d=self.head_k_dim)
        g_slow = rearrange(g_slow, "... (h d) -> ... h d", d=self.head_k_dim)

        if self.num_v_heads > self.num_heads:
            q, k, g_fast, g_slow = (
                repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads)
                for x in (q, k, g_fast, g_slow)
            )
            beta_fast = repeat(beta_fast, "... h -> ... (h g)", g=self.num_v_heads // self.num_heads)
            beta_slow = repeat(beta_slow, "... h -> ... (h g)", g=self.num_v_heads // self.num_heads)
            lam = repeat(lam, "... h -> ... (h g)", g=self.num_v_heads // self.num_heads)

        if self.allow_neg_eigval:
            beta_fast = beta_fast * 2.0
            beta_slow = beta_slow * 2.0

        if mode == "chunk":
            o_f, recurrent_fast = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g_fast,
                beta=beta_fast,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_fast,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                use_gate_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
            o_s, recurrent_slow = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g_slow,
                beta=beta_slow,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_slow,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                use_gate_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        elif mode == "fused_recurrent":
            gf = fused_kda_gate(g=g_fast, A_log=self.A_log, dt_bias=self.dt_bias)
            gs = fused_kda_gate(g=g_slow, A_log=self.A_log, dt_bias=self.dt_bias)
            o_f, recurrent_fast = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=gf,
                beta=beta_fast,
                initial_state=recurrent_fast,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
            )
            o_s, recurrent_slow = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=gs,
                beta=beta_slow,
                initial_state=recurrent_slow,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        o = o_f * lam.unsqueeze(-1) + o_s * (1.0 - lam).unsqueeze(-1)

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=(recurrent_fast, recurrent_slow),
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values

