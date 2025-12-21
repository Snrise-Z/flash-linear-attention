# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Multi-Scale KDA (MSKDA) - KDA with grouped/per-channel A parameters for multi-scale memory

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

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


def compute_log_spaced_timescales(
    num_groups: int,
    tau_min: float = 1.0,
    tau_max: float = 1000.0,
) -> torch.Tensor:
    """
    Compute log-spaced time constants for multi-scale memory.
    
    τ_g = τ_min * (τ_max / τ_min)^(g / (G-1))  for g = 0, 1, ..., G-1
    
    Then A_log_g = log(1/τ_g) = -log(τ_g)
    
    Args:
        num_groups: Number of groups (G)
        tau_min: Minimum time constant (fastest decay)
        tau_max: Maximum time constant (slowest decay)
    
    Returns:
        A_log tensor of shape [num_groups] with log-spaced values
    """
    if num_groups == 1:
        # Single group: use geometric mean
        tau = math.sqrt(tau_min * tau_max)
        return torch.tensor([math.log(1.0 / tau)], dtype=torch.float32)
    
    # Log-uniform spacing: τ_g = τ_min * (τ_max/τ_min)^(g/(G-1))
    exponents = torch.linspace(0, 1, num_groups, dtype=torch.float32)
    taus = tau_min * (tau_max / tau_min) ** exponents  # [G]
    
    # A_log = log(rate) = log(1/τ) = -log(τ)
    # But we want exp(A_log) to be the rate, so A_log = log(1/τ)
    A_log = torch.log(1.0 / taus)
    
    return A_log


def multiscale_gate_forward(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    num_groups: int = 1,
) -> torch.Tensor:
    """
    Compute multi-scale gate with grouped A parameters.
    
    g = -exp(A_log_{h,g}) * softplus(g + dt_bias)
    
    Args:
        g: Input tensor of shape [..., H, K]
        A_log: Parameter tensor of shape [H, G] where G is num_groups
        dt_bias: Optional bias tensor of shape [H * K]
        num_groups: Number of groups per head
    
    Returns:
        Output tensor of shape [..., H, K]
    """
    H, K = g.shape[-2:]
    g = g.float()
    
    if dt_bias is not None:
        g = g + dt_bias.view(H, K)
    
    # Apply softplus
    g_softplus = F.softplus(g)  # [..., H, K]
    
    # Compute -exp(A_log) * softplus(g)
    # A_log has shape [H, G], need to broadcast to [H, K]
    # Each group covers K // G consecutive channels
    A_exp = A_log.float().exp()  # [H, G]
    
    if num_groups == K:
        # Per-channel A: A_log is [H, K]
        A_exp_broadcast = A_exp  # [H, K]
    elif num_groups == 1:
        # Per-head A: A_log is [H, 1] or [H]
        A_exp_broadcast = A_exp.view(H, 1).expand(H, K)  # [H, K]
    else:
        # Grouped A: A_log is [H, G], need to repeat each group K//G times
        channels_per_group = K // num_groups
        A_exp_broadcast = A_exp.repeat_interleave(channels_per_group, dim=-1)  # [H, K]
        # Handle remainder if K is not divisible by G
        if A_exp_broadcast.shape[-1] < K:
            remainder = K - A_exp_broadcast.shape[-1]
            A_exp_broadcast = torch.cat([
                A_exp_broadcast,
                A_exp[:, -1:].expand(H, remainder)
            ], dim=-1)
    
    g_out = -A_exp_broadcast * g_softplus
    
    return g_out


class MultiScaleKimiDeltaAttention(nn.Module):
    """
    Multi-Scale Kimi Delta Attention (MSKDA) layer implementation.
    
    Key improvement over KDA: A_log is extended from per-head [H] to per-group [H, G]
    or per-channel [H, K], enabling multi-scale memory with different time constants
    across channels. This allows the model to learn both fast-decaying (short-term)
    and slow-decaying (long-term) memory patterns.
    
    Mathematical formulation:
        g_{t,h,d} = -exp(A_{h,g(d)}) * softplus(u_{t,h,d} + b_{h,d})
        
    where g(d) maps channel d to its group index.
    
    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 128.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        num_v_heads (int, Optional):
            The number of heads for the value projection. Default: `None`.
        num_a_groups (int, Optional):
            Number of groups for A parameter per head. Default: head_dim (per-channel).
            Set to 1 for per-head A (original KDA behavior).
            Set to K for per-channel A (maximum expressiveness).
            Set to G for grouped A (balanced).
        tau_min (float, Optional):
            Minimum time constant for log-spaced initialization. Default: 1.0.
        tau_max (float, Optional):
            Maximum time constant for log-spaced initialization. Default: 1000.0.
        mode (str, Optional):
            Which kernel to use. Default: `chunk`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`.
        use_tanh_beta (bool, Optional):
            Use 1+tanh parameterization for beta. Default: `False`.
        conv_size (int, Optional):
            The kernel size of the short convolution. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int = None,
        num_a_groups: int = None,  # Default to head_dim (per-channel)
        tau_min: float = 1.0,
        tau_max: float = 1000.0,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        use_tanh_beta: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> MultiScaleKimiDeltaAttention:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.use_tanh_beta = use_tanh_beta
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        
        # Multi-scale A parameters
        # Default: per-channel (num_a_groups = head_dim)
        self.num_a_groups = num_a_groups if num_a_groups is not None else head_dim
        self.tau_min = tau_min
        self.tau_max = tau_max

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )
        
        # Validate num_a_groups
        if self.num_a_groups < 1:
            raise ValueError(f"num_a_groups must be at least 1, got {self.num_a_groups}")
        if self.num_a_groups > head_dim:
            raise ValueError(f"num_a_groups ({self.num_a_groups}) cannot exceed head_dim ({head_dim})")
        
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

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

        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # Multi-scale A_log: shape [H, G] instead of [H]
        # Initialize with log-spaced time constants
        A_log_init = compute_log_spaced_timescales(
            num_groups=self.num_a_groups,
            tau_min=tau_min,
            tau_max=tau_max,
        )  # [G]
        # Expand to [H, G]
        A_log_init = A_log_init.unsqueeze(0).expand(self.num_heads, -1).clone()
        self.A_log = nn.Parameter(A_log_init)
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
        # Handle attention mask normalization
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
        # change to inference mode.
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

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
        beta_raw = self.b_proj(hidden_states)
        
        # Beta parameterization
        if self.use_tanh_beta:
            # β = 1 + tanh(β_raw) ∈ (0, 2)
            # This makes crossing β=1 (for negative eigenvalues) more natural
            beta = 1.0 + torch.tanh(beta_raw)
        else:
            beta = beta_raw.sigmoid()
            if self.allow_neg_eigval:
                beta = beta * 2.0

        q, k, g_raw = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k, g_raw))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        # Compute multi-scale gate outside kernel
        # g_raw: [..., H, K], A_log: [H, G], dt_bias: [H*K]
        g = multiscale_gate_forward(
            g=g_raw,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_groups=self.num_a_groups,
        )
        g = g.to(g_raw.dtype)

        # for multi-value attention, we repeat the inputs for simplicity.
        if self.num_v_heads > self.num_heads:
            q, k, g = (repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads) for x in (q, k, g))
            beta = repeat(beta, "... h -> ... (h g)", g=self.num_v_heads // self.num_heads)

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "chunk":
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=None,  # Gate already computed
                dt_bias=None,  # Gate already computed
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,  # Gate computed outside
                cu_seqlens=cu_seqlens,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

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
