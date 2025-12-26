# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Multi-Scale KDA (MSKDA) Configuration

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class MSKDAConfig(PretrainedConfig):
    """
    Configuration class for Multi-Scale KDA (MSKDA).
    
    MSKDA extends KDA by introducing grouped/per-channel A parameters for multi-scale
    memory. This enables the model to learn different time constants across channels,
    with log-spaced initialization covering a wide range of time scales.
    
    Key differences from KDA:
        - num_a_groups: Number of groups for A parameter (default: head_dim for per-channel)
        - tau_min, tau_max: Time constant range for log-spaced initialization
        - use_tanh_beta: Option to use 1+tanh parameterization for beta
    """
    
    model_type = "mskda"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        use_tanh_beta: bool = False,
        conv_size: int = 4,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        # Multi-scale A parameters
        num_a_groups: int | None = None,  # Default: head_dim (per-channel)
        tau_min: float = 1.0,
        tau_max: float = 1000.0,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 24,
        norm_eps: float = 1e-6,
        attn: dict | None = None,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        **kwargs,
    ):
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size
        self.allow_neg_eigval = allow_neg_eigval
        self.use_tanh_beta = use_tanh_beta
        
        # Multi-scale parameters
        # Default num_a_groups to head_dim for per-channel A
        self.num_a_groups = num_a_groups if num_a_groups is not None else head_dim
        self.tau_min = tau_min
        self.tau_max = tau_max

        if attn is not None:
            if not isinstance(attn, dict):
                raise ValueError("attn must be a dictionary")
            if "layers" not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if "num_heads" not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn["num_kv_heads"] = attn.get("num_kv_heads", attn["num_heads"])
            attn["qkv_bias"] = attn.get("qkv_bias", False)
            attn["window_size"] = attn.get("window_size", None)
            attn["rope_theta"] = attn.get("rope_theta", 10000.0)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
