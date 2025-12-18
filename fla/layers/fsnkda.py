# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING

from fla.layers.fskda import FastSlowSurpriseKimiDeltaAttention

if TYPE_CHECKING:
    import torch
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class FastSlowSurpriseNormalizedKimiDeltaAttention(FastSlowSurpriseKimiDeltaAttention):
    """
    Fast/Slow Surprise-aware Normalized Kimi Delta Attention (FSNKDA).

    This layer is a thin wrapper around the Fast/Slow Surprise-aware KDA (FSKDA) layer
    that enforces key-norm beta normalization (NKDA-style).
    """

    def __init__(self, *args, **kwargs):
        kwargs["use_beta_norm"] = True
        kwargs["use_qk_l2norm_in_kernel"] = False
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        past_key_values: "Cache | None" = None,
        use_cache: "bool | None" = False,
        output_attentions: "bool | None" = False,
        **kwargs: "Unpack[dict]",
    ):
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

