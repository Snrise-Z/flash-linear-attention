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
from fla.ops.kda import chunk_kda_rank_r_microstep, fused_recurrent_kda_rank_r_microstep
from fla.ops.kda.gate import fused_kda_gate

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class MicrostepKimiDeltaAttention(nn.Module):
    """
    MKDA (Micro-step KDA): a rank-r approximation implemented as r serial rank-1 updates per token.

    This layer expands the sequence internally from length T to T*r and reuses existing KDA kernels
    via `chunk_kda_rank_r_microstep` / `fused_recurrent_kda_rank_r_microstep`.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        mode: str = "chunk",
        micro_rank: int = 4,
        micro_fill_g_raw: float = -1.0e4,
        micro_readout_mode: str = "mix",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> MicrostepKimiDeltaAttention:
        super().__init__()

        if micro_rank <= 0:
            raise ValueError(f"micro_rank must be positive, got {micro_rank}.")

        self.mode = mode
        self.micro_rank = micro_rank
        self.micro_fill_g_raw = float(micro_fill_g_raw)
        self.micro_readout_mode = str(micro_readout_mode)
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
        if mode not in ["chunk", "fused_recurrent"]:
            raise ValueError(f"Not supported mode `{mode}`.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim * micro_rank, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim * micro_rank, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim * micro_rank,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim * micro_rank,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        # gate is still per K (no rank), so keep the original shapes
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_heads * micro_rank, bias=False)

        # Micro-step readout mixing weights (per head, over rank dimension).
        # We parameterize as logits and use softmax during forward for stability.
        # Initialize to "mostly last step" to match the previous behavior.
        mix_heads = self.num_v_heads
        init = torch.full((mix_heads, micro_rank), -8.0, dtype=torch.float32)
        init[:, -1] = 0.0
        self.micro_readout_logits = nn.Parameter(init)
        self.micro_readout_logits._no_weight_decay = True

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
        mkda_debug = kwargs.pop("mkda_debug", None)
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

        g_raw = self.f_proj(hidden_states)
        beta = self.b_proj(hidden_states).sigmoid()

        q = rearrange(q, "... (h d) -> ... h d", h=self.num_heads, d=self.head_k_dim)
        g_raw = rearrange(g_raw, "... (h d) -> ... h d", h=self.num_heads, d=self.head_k_dim)
        k = rearrange(k, "... (h r d) -> ... h r d", h=self.num_heads, r=self.micro_rank, d=self.head_k_dim)
        v = rearrange(v, "... (h r d) -> ... h r d", h=self.num_v_heads, r=self.micro_rank, d=self.head_v_dim)
        beta = rearrange(beta, "... (h r) -> ... h r", h=self.num_heads, r=self.micro_rank)

        if self.num_v_heads > self.num_heads:
            group = self.num_v_heads // self.num_heads
            q = repeat(q, "... h d -> ... (h g) d", g=group)
            g_raw = repeat(g_raw, "... h d -> ... (h g) d", g=group)
            k = repeat(k, "... h r d -> ... (h g) r d", g=group)
            beta = repeat(beta, "... h r -> ... (h g) r", g=group)

        if self.allow_neg_eigval:
            beta = beta * 2.0

        if mkda_debug is not None:
            with torch.no_grad():
                k_fp32 = k.to(torch.float32)  # [B,T,H,R,K]
                gram = torch.matmul(k_fp32, k_fp32.transpose(-1, -2))  # [B,T,H,R,R]
                R = gram.shape[-1]
                eye = torch.eye(R, device=gram.device, dtype=torch.bool).view(1, 1, 1, R, R)
                off = ~eye

                gram_off_abs_mean = gram.abs().masked_select(off).mean().item()
                diag = gram.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)  # [B,T,H,R]
                denom = (diag[..., :, None] * diag[..., None, :]).sqrt()
                cos = (gram / denom).clamp(min=-1.0, max=1.0)
                cos_off_abs_mean = cos.abs().masked_select(off).mean().item()

                beta_fp32 = beta.to(torch.float32)
                beta_min = beta_fp32.min().item()
                beta_max = beta_fp32.max().item()
                beta_mean = beta_fp32.mean().item()
                beta_rms = beta_fp32.square().mean().sqrt().item()
                beta_mean_per_r = beta_fp32.mean(dim=(0, 1, 2)).tolist()
                beta_rms_per_r = beta_fp32.square().mean(dim=(0, 1, 2)).sqrt().tolist()

                mkda_debug.append(
                    {
                        "layer_idx": self.layer_idx,
                        "micro_rank": self.micro_rank,
                        "k_gram_offdiag_abs_mean": gram_off_abs_mean,
                        "k_cos_offdiag_abs_mean": cos_off_abs_mean,
                        "beta_min": beta_min,
                        "beta_max": beta_max,
                        "beta_mean": beta_mean,
                        "beta_rms": beta_rms,
                        "beta_mean_per_r": beta_mean_per_r,
                        "beta_rms_per_r": beta_rms_per_r,
                    }
                )

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "chunk":
            # Gate refinement: compute log-decay in PyTorch, then force "no gate" on non-first micro-steps
            # by filling g_micro with exact zeros (see microstep expander). This avoids relying on an
            # extreme raw-gate fill value passing through the kernel-side nonlinearity.
            g = fused_kda_gate(g=g_raw, A_log=self.A_log, dt_bias=self.dt_bias)
            o_micro, recurrent_state = chunk_kda_rank_r_microstep(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                micro_readout="all" if self.micro_readout_mode == "mix" else "last",
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
                cu_seqlens=cu_seqlens,
            )
        elif mode == "fused_recurrent":
            g = fused_kda_gate(g=g_raw, A_log=self.A_log, dt_bias=self.dt_bias)
            o_micro, recurrent_state = fused_recurrent_kda_rank_r_microstep(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                micro_readout="all" if self.micro_readout_mode == "mix" else "last",
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if self.micro_readout_mode == "mix":
            # Mix micro-step outputs: o_micro is [B,T,R,H,Dv] -> o is [B,T,H,Dv]
            gamma = torch.softmax(self.micro_readout_logits, dim=-1).to(dtype=o_micro.dtype, device=o_micro.device)
            o = (o_micro * gamma.view(1, 1, self.micro_rank, -1, 1)).sum(dim=2)
        elif self.micro_readout_mode == "last":
            # o_micro is already [B,T,H,Dv]
            o = o_micro
        else:
            raise ValueError(f"Unsupported micro_readout_mode={self.micro_readout_mode!r}; expected 'mix' or 'last'.")

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
