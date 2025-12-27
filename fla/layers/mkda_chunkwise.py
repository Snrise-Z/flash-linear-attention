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
from fla.ops.kda import chunk_kda_rank_n
from fla.ops.mkda import mkda_chunkwise_parallel
from fla.ops.kda.gate import fused_kda_gate

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class ChunkwiseMultiKeyDeltaAttention(nn.Module):
    """
    True MKDA (Multi-Key Delta Attention) implemented with a chunkwise-parallel algorithm:
      - inside a chunk, solve a block unit-lower-triangular system once and use a cumsum
      - across chunks, carry the recurrent state S (like other delta-rule methods)

    For `num_keys<=4` on CUDA, this uses the exact Triton MKDA kernel (`chunk_kda_rank_n`).
    Otherwise it falls back to a reference PyTorch implementation (`mkda_chunkwise_parallel`).
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        num_keys: int = 4,
        chunk_size: int = 64,
        rank_mix: str = "none",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> ChunkwiseMultiKeyDeltaAttention:
        super().__init__()

        if num_keys <= 0:
            raise ValueError(f"num_keys must be positive, got {num_keys}.")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

        self.num_keys = int(num_keys)
        self.chunk_size = int(chunk_size)
        self.rank_mix = str(rank_mix)
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

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

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim * self.num_keys, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim * self.num_keys, bias=False)

        if self.rank_mix not in ("none", "kv"):
            raise ValueError(f"rank_mix must be one of {{'none','kv'}}, got {self.rank_mix!r}.")
        if self.rank_mix == "kv" and self.num_keys > 1:
            self.rank_mix_k = nn.Linear(self.num_keys, self.num_keys, bias=False)
            self.rank_mix_v = nn.Linear(self.num_keys, self.num_keys, bias=False)
            nn.init.eye_(self.rank_mix_k.weight)
            nn.init.eye_(self.rank_mix_v.weight)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim * self.num_keys,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim * self.num_keys,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_heads * self.num_keys, bias=False)

        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True

        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=True),
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
        if output_attentions:
            raise ValueError("MKDA does not support output_attentions.")

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
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens")
        use_triton_mkda = self.num_keys <= 4 and hidden_states.is_cuda

        indices = None
        if attention_mask is not None:
            if not use_triton_mkda:
                raise ValueError("attention_mask/padded batches require num_keys<=4 CUDA (Triton MKDA path).")
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        elif cu_seqlens is not None and not use_triton_mkda:
            raise ValueError("cu_seqlens-packed inputs require num_keys<=4 CUDA (Triton MKDA path).")

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

        g_raw = self.f_proj(hidden_states)
        beta = self.b_proj(hidden_states).sigmoid()

        q = rearrange(q, "... (h d) -> ... h d", h=self.num_heads, d=self.head_k_dim)
        g_raw = rearrange(g_raw, "... (h d) -> ... h d", h=self.num_heads, d=self.head_k_dim)
        k = rearrange(k, "... (h r d) -> ... h r d", h=self.num_heads, r=self.num_keys, d=self.head_k_dim)
        v = rearrange(v, "... (h r d) -> ... h r d", h=self.num_v_heads, r=self.num_keys, d=self.head_v_dim)
        beta = rearrange(beta, "... (h r) -> ... h r", h=self.num_heads, r=self.num_keys)

        if self.num_v_heads > self.num_heads:
            group = self.num_v_heads // self.num_heads
            q = repeat(q, "... h d -> ... (h g) d", g=group)
            g_raw = repeat(g_raw, "... h d -> ... (h g) d", g=group)
            k = repeat(k, "... h r d -> ... (h g) r d", g=group)
            beta = repeat(beta, "... h r -> ... (h g) r", g=group)

        if self.allow_neg_eigval:
            beta = beta * 2.0

        if self.rank_mix == "kv" and self.num_keys > 1:
            k = self.rank_mix_k(k.transpose(-1, -2)).transpose(-1, -2)
            v = self.rank_mix_v(v.transpose(-1, -2)).transpose(-1, -2)

        # Match KDA's "use_qk_l2norm_in_kernel=True" behavior explicitly.
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        log_alpha = fused_kda_gate(g=g_raw, A_log=self.A_log, dt_bias=self.dt_bias).to(torch.float32)

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if use_triton_mkda:
            o, recurrent_state = chunk_kda_rank_n(
                q=q,
                k=k,
                v=v,
                log_alpha=log_alpha,
                beta=beta,
                initial_state=recurrent_state,
                scale=self.head_k_dim**-0.5,
                output_final_state=bool(use_cache),
                cu_seqlens=cu_seqlens,
                chunk_size=self.chunk_size,
            )
        else:
            o, recurrent_state = mkda_chunkwise_parallel(
                q=q,
                k=k,
                v=v,
                log_alpha=log_alpha,
                beta=beta,
                initial_state=recurrent_state,
                scale=self.head_k_dim**-0.5,
                chunk_size=self.chunk_size,
                output_final_state=bool(use_cache),
            )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
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
