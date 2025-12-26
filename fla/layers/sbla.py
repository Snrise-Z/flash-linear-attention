# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# SBLA: Sqrt-Block Landmark Attention (O(L^1.5))
# - FlashAttention (via PyTorch SDPA) for local sqrt blocks (no attn_mask, optional causal)
# - Memory-efficient global paths:
#     * token -> landmark : query-chunked (float32 softmax)
#     * landmark -> token refine : key-streaming (online softmax), exact
# - KV-cache incremental decoding (prefill + forward_step), same weights
# - Multi-landmarks per block + learnable attention pooling
# - Landmark self-attention (block-level causal mask, allows intra-block free mixing)
# - Optional landmark->token refinement
# - Per-head gating fusion
# - RoPE (rotary positional embeddings)
#
# Reference implementation by ChatGPT (GPT-5.2 Pro)
# Adapted for fla library compatibility

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Callable, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from fla.models.utils import Cache


# ============================================================
# RoPE (Rotary Positional Embedding)
# ============================================================

class RotaryEmbedding(nn.Module):
    """
    Cache cos/sin for positions [0..seq_len-1].
    We rotate the first `rotary_dim` dims (must be even).
    """
    def __init__(self, rotary_dim: int, base: int = 10000):
        super().__init__()
        rotary_dim = int(rotary_dim)
        assert rotary_dim % 2 == 0, "rotary_dim must be even"
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._cached_seq_len: int = 0
        self._cached_device: Optional[torch.device] = None
        self._cached_dtype: Optional[torch.dtype] = None

    @torch.no_grad()
    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))  # [L, dim/2]
        self._cos_cached = freqs.cos().to(dtype=dtype)
        self._sin_cached = freqs.sin().to(dtype=dtype)
        self._cached_seq_len = int(seq_len)
        self._cached_device = device
        self._cached_dtype = dtype

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if (self._cos_cached is None
            or self._sin_cached is None
            or self._cached_seq_len < seq_len
            or self._cached_device != device
            or self._cached_dtype != dtype):
            self._build_cache(seq_len, device, dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """
    Apply RoPE to x[..., :rotary_dim].
    x: [B,H,L,Dh] (or broadcastable)
    cos/sin: broadcastable to [L, rotary_dim/2]
    """
    if rotary_dim == 0:
        return x

    x1 = x[..., :rotary_dim]
    x2 = x[..., rotary_dim:]

    x1 = x1.view(*x1.shape[:-1], rotary_dim // 2, 2)
    x_even = x1[..., 0]
    x_odd = x1[..., 1]

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
    return torch.cat((out, x2), dim=-1)


# ============================================================
# Stable masked softmax (float32)
# ============================================================

def masked_softmax(scores: torch.Tensor, valid: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    """
    scores: any shape
    valid: bool mask broadcastable to scores, True=keep
    Softmax in float32 for stability; fully-masked row -> all zeros.
    """
    scores_f = scores.float()
    neg_inf = -float("inf")
    scores_f = scores_f.masked_fill(~valid, neg_inf)

    max_scores = scores_f.max(dim=dim, keepdim=True).values
    max_scores = max_scores.masked_fill(~torch.isfinite(max_scores), 0.0)

    exp = (scores_f - max_scores).exp()
    exp = exp * valid.to(exp.dtype)

    denom = exp.sum(dim=dim, keepdim=True).clamp_min(eps)
    out = exp / denom
    return out.to(dtype=scores.dtype)


# ============================================================
# Query-chunked exact attention (float32 softmax)
# ============================================================

def attention_query_chunked(
    q: torch.Tensor,  # [B,H,Lq,Dh]
    k: torch.Tensor,  # [B,H,Lk,Dh]
    v: torch.Tensor,  # [B,H,Lk,Dh]
    key_valid: Optional[torch.Tensor],  # [B,Lk] bool True=valid
    mask_fn: Optional[Callable[[int, int], Optional[torch.Tensor]]],  # returns [Lq_chunk,Lk] bool True=mask out
    *,
    chunk_size_q: int = 2048,
    attn_dropout: float = 0.0,
    training: bool = False,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Exact attention. Softmax in float32. Computed in chunks over query length to cap memory.
    """
    B, H, Lq, Dh = q.shape
    _, _, Lk, _ = k.shape
    scale = Dh ** -0.5
    neg_inf = -float("inf")

    out = torch.empty((B, H, Lq, Dh), device=q.device, dtype=q.dtype)
    k_t = k.transpose(-2, -1)  # [B,H,Dh,Lk]
    v_f = v.float()

    for s in range(0, Lq, chunk_size_q):
        e = min(s + chunk_size_q, Lq)
        q_chunk = q[:, :, s:e, :]                           # [B,H,chunk,Dh]
        scores = torch.matmul(q_chunk, k_t) * scale          # [B,H,chunk,Lk]
        scores_f = scores.float()

        if key_valid is not None:
            scores_f = scores_f.masked_fill(~key_valid[:, None, None, :], neg_inf)

        cmask = None
        if mask_fn is not None:
            cmask = mask_fn(s, e)
            if cmask is not None:
                scores_f = scores_f.masked_fill(cmask[None, None, :, :], neg_inf)

        max_scores = scores_f.max(dim=-1, keepdim=True).values
        max_scores = max_scores.masked_fill(~torch.isfinite(max_scores), 0.0)

        attn = (scores_f - max_scores).exp()
        if key_valid is not None:
            attn = attn * key_valid[:, None, None, :].to(attn.dtype)
        if cmask is not None:
            attn = attn * (~cmask)[None, None, :, :].to(attn.dtype)

        denom = attn.sum(dim=-1, keepdim=True).clamp_min(eps)
        attn = attn / denom

        if attn_dropout > 0.0 and training:
            attn = F.dropout(attn, p=attn_dropout)

        out_chunk = torch.matmul(attn, v_f)                  # float32 accumulate
        out[:, :, s:e, :] = out_chunk.to(dtype=q.dtype)

    return out


# ============================================================
# Key-streaming exact attention (FlashAttention-style online softmax)
# ============================================================

def attention_key_streaming(
    q: torch.Tensor,  # [B,H,Lq,Dh]
    k: torch.Tensor,  # [B,H,Lk,Dh]
    v: torch.Tensor,  # [B,H,Lk,Dh]
    key_valid: Optional[torch.Tensor],  # [B,Lk] bool True=valid
    mask_fn: Optional[Callable[[int, int], Optional[torch.Tensor]]],  # returns [Lq,Kchunk] bool True=mask out
    *,
    chunk_size_k: int = 2048,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Exact attention, streaming over key length to avoid materializing [Lq,Lk].
    Dropout is intentionally not supported (use eval / attn_dropout=0).
    """
    B, H, Lq, Dh = q.shape
    _, _, Lk, _ = k.shape
    scale = Dh ** -0.5
    neg_inf = -float("inf")

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    m = torch.full((B, H, Lq, 1), neg_inf, device=q.device, dtype=torch.float32)
    l = torch.zeros((B, H, Lq, 1), device=q.device, dtype=torch.float32)
    out = torch.zeros((B, H, Lq, Dh), device=q.device, dtype=torch.float32)

    for ks in range(0, Lk, chunk_size_k):
        ke = min(ks + chunk_size_k, Lk)
        k_chunk = k_f[:, :, ks:ke, :]                             # [B,H,Kc,Dh]
        v_chunk = v_f[:, :, ks:ke, :]

        scores = torch.matmul(q_f, k_chunk.transpose(-2, -1)) * scale  # [B,H,Lq,Kc]

        if key_valid is not None:
            scores = scores.masked_fill(~key_valid[:, None, None, ks:ke], neg_inf)

        cmask = None
        if mask_fn is not None:
            cmask = mask_fn(ks, ke)                                # [Lq,Kc]
            if cmask is not None:
                scores = scores.masked_fill(cmask[None, None, :, :], neg_inf)

        block_max = scores.max(dim=-1, keepdim=True).values         # [B,H,Lq,1]
        m_new = torch.maximum(m, block_max)

        finite = torch.isfinite(m_new)
        exp_m = torch.exp(m - m_new)
        exp_m = torch.where(finite, exp_m, torch.zeros_like(exp_m))

        scores_shift = scores - m_new
        scores_shift = torch.where(finite, scores_shift, torch.full_like(scores_shift, neg_inf))
        exp_scores = torch.exp(scores_shift)

        l_new = l * exp_m + exp_scores.sum(dim=-1, keepdim=True)
        out = out * exp_m + torch.matmul(exp_scores, v_chunk)

        m, l = m_new, l_new

    out = out / l.clamp_min(eps)
    return out.to(dtype=q.dtype)


# ============================================================
# SBLA Attention + KV-cache
# ============================================================

class SBLAAttention(nn.Module):
    """
    SBLA: Local sqrt-block attention + token<->landmark global exchange.

    Features:
      - M landmarks per block via learnable attention pooling
      - Landmark self-attn (block-level causal, intra-block free mixing)
      - Optional landmark->token refinement (Perceiver-like) to strengthen landmarks
      - token->landmark global attention (strict previous blocks in causal mode)
      - per-head gating fusion
      - RoPE
      - KV-cache incremental decoding API: init_cache / prefill / forward_step

    Forward supports right-padding key_padding_mask (pads at the end).
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        *,
        num_landmarks: int = 4,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_rope: bool = True,
        rope_theta: float = 10000.,
        rope_dim: Optional[int] = None,
        use_landmark_self_attn: bool = True,
        use_landmark_to_token: bool = True,
        gating: bool = True,
        # memory knobs
        token_to_landmark_q_chunk: int = 2048,
        lm_to_token_k_chunk: int = 2048,
        # block sizing
        fixed_block_size: Optional[int] = None,  # strongly recommended for cache consistency
        eps: float = 1e-9,
        bias: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert num_landmarks >= 1

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.num_landmarks = int(num_landmarks)

        self.attn_dropout = float(attn_dropout)
        self.proj_dropout = float(proj_dropout)
        self.use_landmark_self_attn = bool(use_landmark_self_attn)
        self.use_landmark_to_token = bool(use_landmark_to_token)
        self.gating = bool(gating)

        self.token_to_landmark_q_chunk = int(token_to_landmark_q_chunk)
        self.lm_to_token_k_chunk = int(lm_to_token_k_chunk)
        self.eps = float(eps)

        self.fixed_block_size = int(fixed_block_size) if fixed_block_size is not None else None
        if self.fixed_block_size is not None:
            assert self.fixed_block_size > 0

        self.layer_idx = layer_idx

        # Token QKV
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        # Learnable pooling queries (M queries)
        self.pool_queries = nn.Parameter(torch.randn(num_landmarks, hidden_size) / math.sqrt(hidden_size))

        # Landmark self-attn
        self.lm_qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.lm_out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Landmark -> token refinement
        self.lm_to_tok_q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.lm_to_tok_out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Landmark KV memory for token->landmark
        self.lm_kv_proj = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)

        # Output
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.proj_drop = nn.Dropout(proj_dropout)

        # Per-head gating
        if self.gating:
            self.gate_proj = nn.Linear(hidden_size, num_heads, bias=True)

        # RoPE
        self.use_rope = bool(use_rope)
        if self.use_rope:
            rd = int(rope_dim) if rope_dim is not None else min(64, self.head_dim)
            rd = rd - (rd % 2)
            rd = max(0, min(rd, self.head_dim))
            self.rope_dim = int(rd)
            self.rope = RotaryEmbedding(self.rope_dim, base=int(rope_theta)) if self.rope_dim > 0 else None
        else:
            self.rope_dim = 0
            self.rope = None

    # --------------------
    # helpers
    # --------------------
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,D] -> [B,H,L,Dh]
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,H,L,Dh] -> [B,L,D]
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def _choose_block_size(self, L: int) -> int:
        if self.fixed_block_size is not None:
            return self.fixed_block_size
        return int(math.ceil(math.sqrt(L))) if L > 1 else 1

    def _get_lm_offsets(self, block_size: int, device: torch.device) -> torch.Tensor:
        # positions of M landmarks inside a block
        offsets = torch.floor(
            (torch.arange(self.num_landmarks, device=device, dtype=torch.float32) + 0.5)
            * block_size / self.num_landmarks
        ).to(torch.long).clamp(0, block_size - 1)
        return offsets

    def _rope_apply_qk(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if (not self.use_rope) or self.rope_dim == 0:
            return q, k
        assert self.rope is not None
        max_pos = int(pos.max().item() + 1)
        cos, sin = self.rope.get_cos_sin(max_pos, device=q.device, dtype=q.dtype)
        return (
            apply_rope(q, cos[pos], sin[pos], self.rope_dim),
            apply_rope(k, cos[pos], sin[pos], self.rope_dim),
        )

    def _rope_apply_q(self, q: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        if (not self.use_rope) or self.rope_dim == 0:
            return q
        assert self.rope is not None
        max_pos = int(pos.max().item() + 1)
        cos, sin = self.rope.get_cos_sin(max_pos, device=q.device, dtype=q.dtype)
        return apply_rope(q, cos[pos], sin[pos], self.rope_dim)

    def _rope_apply_k(self, k: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        if (not self.use_rope) or self.rope_dim == 0:
            return k
        assert self.rope is not None
        max_pos = int(pos.max().item() + 1)
        cos, sin = self.rope.get_cos_sin(max_pos, device=k.device, dtype=k.dtype)
        return apply_rope(k, cos[pos], sin[pos], self.rope_dim)

    def _rope_apply_single(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        # x: [B,H,1,Dh]
        if (not self.use_rope) or self.rope_dim == 0:
            return x
        assert self.rope is not None
        cos, sin = self.rope.get_cos_sin(pos + 1, device=x.device, dtype=x.dtype)
        return apply_rope(x, cos[pos:pos+1], sin[pos:pos+1], self.rope_dim)

    # --------------------
    # vectorized local block attention
    # --------------------
    def _local_attention_vectorized(
        self,
        q_tok: torch.Tensor,  # [B,H,Lpad,Dh]
        k_tok: torch.Tensor,  # [B,H,Lpad,Dh]
        v_tok: torch.Tensor,  # [B,H,Lpad,Dh]
        lengths: torch.Tensor,  # [B] actual lengths (<= Lpad)
        block_size: int,
        is_causal: bool,
        dropout_p: float,
    ) -> torch.Tensor:
        """
        Compute local attention inside each block.
        Vectorized across ALL full blocks in the batch -> single SDPA call (Flash if CUDA).
        Tail (partial) block per sample handled separately (<=B SDPA calls).
        """
        B, H, Lpad, Dh = q_tok.shape
        nb = Lpad // block_size
        device = q_tok.device

        local_out = torch.zeros((B, H, Lpad, Dh), device=device, dtype=q_tok.dtype)

        # reshape into blocks
        q_blk = q_tok.view(B, H, nb, block_size, Dh).permute(0, 2, 1, 3, 4).contiguous()  # [B,nb,H,bs,Dh]
        k_blk = k_tok.view(B, H, nb, block_size, Dh).permute(0, 2, 1, 3, 4).contiguous()
        v_blk = v_tok.view(B, H, nb, block_size, Dh).permute(0, 2, 1, 3, 4).contiguous()

        # which blocks are full (end <= length)
        block_ends = (torch.arange(nb, device=device) + 1) * block_size              # [nb]
        full_mask = block_ends[None, :] <= lengths[:, None]                          # [B,nb] bool
        full_b, full_bi = full_mask.nonzero(as_tuple=True)
        nfull = int(full_b.numel())

        if nfull > 0:
            q_full = q_blk[full_b, full_bi].contiguous()  # [nfull,H,bs,Dh]
            k_full = k_blk[full_b, full_bi].contiguous()
            v_full = v_blk[full_b, full_bi].contiguous()

            out_full = F.scaled_dot_product_attention(
                q_full, k_full, v_full,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )  # [nfull,H,bs,Dh]

            # scatter back
            out_blk = local_out.view(B, H, nb, block_size, Dh).permute(0, 2, 1, 3, 4).contiguous()  # [B,nb,H,bs,Dh]
            out_blk[full_b, full_bi] = out_full
            local_out = out_blk.permute(0, 2, 1, 3, 4).contiguous().view(B, H, Lpad, Dh)

        # tail blocks (partial, at most B)
        tail_len = lengths % block_size
        tail_b = (tail_len > 0).nonzero(as_tuple=True)[0]
        for b in tail_b.tolist():
            tl = int(tail_len[b].item())
            bi = int((lengths[b].item()) // block_size)
            s = bi * block_size
            e = s + tl
            q_t = q_tok[b:b+1, :, s:e, :].contiguous()
            k_t = k_tok[b:b+1, :, s:e, :].contiguous()
            v_t = v_tok[b:b+1, :, s:e, :].contiguous()

            out_t = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )  # [1,H,tl,Dh]
            local_out[b:b+1, :, s:e, :] = out_t

        return local_out

    # --------------------
    # main forward (Flash local + memory-efficient global)
    # --------------------
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B,L,D]
        attention_mask: Optional[torch.Tensor] = None,  # [B,L] bool True=pad
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """
        Full forward.
        - For encoder: is_causal=False
        - For decoder self-attn: is_causal=True (strictly no future leakage)
        
        Args:
            hidden_states: [B, L, D]
            attention_mask: [B, L] bool mask (True = padding/masked token)
            past_key_values: Cache object for incremental decoding (not used in forward, use prefill/forward_step)
            use_cache: Whether to return cache (not used in forward)
            output_attentions: Whether to return attention weights (not supported, returns None)
        
        Returns:
            Tuple of (output, None, past_key_values)
        """
        # For causal decoder (default in fla)
        is_causal = True
        
        B, L, D = hidden_states.shape
        device = hidden_states.device
        if L == 0:
            return hidden_states, None, past_key_values

        # right-padding lengths
        if attention_mask is None:
            lengths = torch.full((B,), L, device=device, dtype=torch.long)
            kpm = torch.zeros((B, L), device=device, dtype=torch.bool)
        else:
            kpm = attention_mask
            lengths = (~kpm).sum(dim=1)  # assumes right-padding

        # block config
        block_size = self._choose_block_size(L)
        nb = int(math.ceil(L / block_size))
        Lpad = nb * block_size

        # pad x and mask
        if Lpad != L:
            xpad = F.pad(hidden_states, (0, 0, 0, Lpad - L), value=0.0)
            kpm = F.pad(kpm, (0, Lpad - L), value=True)
        else:
            xpad = hidden_states

        tok_valid = ~kpm  # [B,Lpad]

        # token qkv
        qkv = self.qkv_proj(xpad)
        q_tok, k_tok, v_tok = qkv.chunk(3, dim=-1)
        q_tok = self._split_heads(q_tok)  # [B,H,Lpad,Dh]
        k_tok = self._split_heads(k_tok)
        v_tok = self._split_heads(v_tok)

        # RoPE for tokens
        if self.use_rope and self.rope_dim > 0:
            pos_tok = torch.arange(Lpad, device=device, dtype=torch.long)
            q_tok, k_tok = self._rope_apply_qk(q_tok, k_tok, pos_tok)

        dropout_p = self.attn_dropout if self.training else 0.0

        # 1) Local blocks (vectorized over full blocks)
        local_out = self._local_attention_vectorized(
            q_tok, k_tok, v_tok,
            lengths=lengths.clamp_max(Lpad),
            block_size=block_size,
            is_causal=is_causal,
            dropout_p=dropout_p,
        )
        local_out = local_out * tok_valid[:, None, :, None].to(local_out.dtype)

        # 2) Build raw landmarks for all blocks (including empty ones, masked)
        x_block = xpad.view(B, nb, block_size, D)        # [B,nb,bs,D]
        blk_valid = tok_valid.view(B, nb, block_size)    # [B,nb,bs]
        blk_has_any = blk_valid.any(dim=-1)              # [B,nb]

        scores_pool = torch.einsum("md,bntd->bnmt", self.pool_queries, x_block) / math.sqrt(D)  # [B,nb,M,bs]
        w_pool = masked_softmax(scores_pool, blk_valid[:, :, None, :], dim=-1, eps=self.eps).float()
        landmarks = torch.einsum("bnmt,bntd->bnmd", w_pool, x_block.float()).to(dtype=hidden_states.dtype)  # [B,nb,M,D]
        landmarks = landmarks.reshape(B, nb * self.num_landmarks, D)                             # [B,Lm,D]

        Lm = nb * self.num_landmarks
        lm_valid = blk_has_any[:, :, None].expand(B, nb, self.num_landmarks).reshape(B, Lm)      # [B,Lm]

        lm_block_idx = torch.arange(nb, device=device).repeat_interleave(self.num_landmarks)    # [Lm]

        if self.use_rope and self.rope_dim > 0:
            offsets = self._get_lm_offsets(block_size, device)
            lm_pos = (torch.arange(nb, device=device, dtype=torch.long) * block_size).repeat_interleave(self.num_landmarks) \
                     + offsets.repeat(nb)
        else:
            lm_pos = None

        # 3) Landmark self-attn (block-level causal)
        if self.use_landmark_self_attn:
            lm_qkv = self.lm_qkv_proj(landmarks)
            q_lm, k_lm, v_lm = lm_qkv.chunk(3, dim=-1)
            q_lm = self._split_heads(q_lm)
            k_lm = self._split_heads(k_lm)
            v_lm = self._split_heads(v_lm)

            if self.use_rope and self.rope_dim > 0 and lm_pos is not None:
                q_lm, k_lm = self._rope_apply_qk(q_lm, k_lm, lm_pos)

            causal_lm = (lm_block_idx[None, :] > lm_block_idx[:, None]) if is_causal else None

            def lm_mask_fn(s: int, e: int) -> Optional[torch.Tensor]:
                if causal_lm is None:
                    return None
                return causal_lm[s:e, :]

            lm_out = attention_query_chunked(
                q_lm, k_lm, v_lm,
                key_valid=lm_valid,
                mask_fn=lm_mask_fn,
                chunk_size_q=1024,
                attn_dropout=dropout_p,
                training=self.training,
                eps=self.eps,
            )  # [B,H,Lm,Dh]
            lm_out = self._merge_heads(lm_out)
            lm_out = self.lm_out_proj(lm_out)
            landmarks = landmarks + self.proj_drop(lm_out)
            landmarks = landmarks * lm_valid[:, :, None].to(landmarks.dtype)

        # 4) Landmark -> token refinement (block-level causal)
        if self.use_landmark_to_token:
            q_ref = self.lm_to_tok_q_proj(landmarks)
            q_ref = self._split_heads(q_ref)
            if self.use_rope and self.rope_dim > 0 and lm_pos is not None:
                q_ref = self._rope_apply_q(q_ref, lm_pos)

            tok_block_idx = torch.arange(Lpad, device=device) // block_size  # [Lpad]

            def lm_to_tok_mask_fn(ks: int, ke: int) -> Optional[torch.Tensor]:
                if not is_causal:
                    return None
                kb = tok_block_idx[ks:ke]  # [Kc]
                # mask future token blocks for each landmark
                return kb[None, :] > lm_block_idx[:, None]  # [Lm,Kc]

            if (self.training and dropout_p > 0.0):
                # dropout for streaming is non-trivial; fallback to query-chunked exact
                full_mask = lm_to_tok_mask_fn(0, Lpad) if is_causal else None

                def qmask(s: int, e: int) -> Optional[torch.Tensor]:
                    if full_mask is None:
                        return None
                    return full_mask[s:e, :]

                ref_out = attention_query_chunked(
                    q_ref, k_tok, v_tok,
                    key_valid=tok_valid,
                    mask_fn=qmask,
                    chunk_size_q=256,
                    attn_dropout=dropout_p,
                    training=True,
                    eps=self.eps,
                )
            else:
                ref_out = attention_key_streaming(
                    q_ref, k_tok, v_tok,
                    key_valid=tok_valid,
                    mask_fn=lm_to_tok_mask_fn if is_causal else None,
                    chunk_size_k=self.lm_to_token_k_chunk,
                    eps=self.eps,
                )

            ref_out = self._merge_heads(ref_out)
            ref_out = self.lm_to_tok_out_proj(ref_out)
            landmarks = landmarks + self.proj_drop(ref_out)
            landmarks = landmarks * lm_valid[:, :, None].to(landmarks.dtype)

        # 5) Token -> landmarks global attention (strict previous blocks if causal)
        lm_kv = self.lm_kv_proj(landmarks)
        k_gl, v_gl = lm_kv.chunk(2, dim=-1)
        k_gl = self._split_heads(k_gl)
        v_gl = self._split_heads(v_gl)

        if self.use_rope and self.rope_dim > 0 and lm_pos is not None:
            k_gl = self._rope_apply_k(k_gl, lm_pos)

        tok_block_idx = torch.arange(Lpad, device=device) // block_size

        if is_causal:
            def tok_to_lm_mask_fn(s: int, e: int) -> Optional[torch.Tensor]:
                qb = tok_block_idx[s:e]
                # strict: mask landmark block >= token block
                return lm_block_idx[None, :] >= qb[:, None]
        else:
            tok_to_lm_mask_fn = None

        global_out = attention_query_chunked(
            q_tok, k_gl, v_gl,
            key_valid=lm_valid,
            mask_fn=tok_to_lm_mask_fn,
            chunk_size_q=self.token_to_landmark_q_chunk,
            attn_dropout=dropout_p,
            training=self.training,
            eps=self.eps,
        )
        global_out = global_out * tok_valid[:, None, :, None].to(global_out.dtype)

        # 6) Fuse local + global
        if self.gating:
            gate = torch.sigmoid(self.gate_proj(xpad))              # [B,Lpad,H]
            gate = gate.transpose(1, 2).unsqueeze(-1)               # [B,H,Lpad,1]
            fused = local_out + gate * global_out
        else:
            fused = local_out + global_out

        y = self._merge_heads(fused)                                # [B,Lpad,D]
        y = self.out_proj(y)
        y = self.proj_drop(y)

        y = y[:, :L, :]
        if attention_mask is not None:
            y = y.masked_fill(attention_mask[:, :L, None], 0.0)
        
        return y, None, past_key_values

    # ============================================================
    # KV-cache API (decoder causal)
    # ============================================================
    @torch.no_grad()
    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """
        Initialize KV-cache buffers for incremental decoding.
        Assumption: all sequences in batch share the same current step `t`.
        """
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype

        block_size = self._choose_block_size(max_seq_len)
        num_blocks_max = int(math.ceil(max_seq_len / block_size))
        max_lm = num_blocks_max * self.num_landmarks

        cache: Dict[str, Any] = {}
        cache["t"] = 0
        cache["max_seq_len"] = int(max_seq_len)
        cache["block_size"] = int(block_size)
        cache["num_blocks_max"] = int(num_blocks_max)
        cache["num_landmarks"] = int(self.num_landmarks)

        # token KV
        cache["tok_k"] = torch.empty((batch_size, self.num_heads, max_seq_len, self.head_dim), device=device, dtype=dtype)
        cache["tok_v"] = torch.empty((batch_size, self.num_heads, max_seq_len, self.head_dim), device=device, dtype=dtype)

        # current partial block raw x
        cache["cur_x"] = torch.empty((batch_size, block_size, self.hidden_size), device=device, dtype=dtype)
        cache["cur_len"] = 0
        cache["cur_block_idx"] = 0

        # landmark self-attn KV cache (K,V from raw pooled landmarks, after RoPE)
        cache["lm_k_sa"] = torch.empty((batch_size, self.num_heads, max_lm, self.head_dim), device=device, dtype=dtype)
        cache["lm_v_sa"] = torch.empty((batch_size, self.num_heads, max_lm, self.head_dim), device=device, dtype=dtype)
        cache["lm_len_sa"] = 0

        # token->landmark KV cache (K,V from final refined landmarks, after RoPE on K)
        cache["lm_k"] = torch.empty((batch_size, self.num_heads, max_lm, self.head_dim), device=device, dtype=dtype)
        cache["lm_v"] = torch.empty((batch_size, self.num_heads, max_lm, self.head_dim), device=device, dtype=dtype)
        cache["lm_len"] = 0

        cache["lm_offsets"] = self._get_lm_offsets(block_size, device=device)  # [M]
        return cache

    @torch.no_grad()
    def prefill(self, x: torch.Tensor, cache: Dict[str, Any]) -> torch.Tensor:
        """
        Causal prefill for decoder:
          - computes outputs y for the prompt
          - fills token KV
          - builds/stores landmarks for COMPLETE blocks only
          - stores incomplete tail tokens into cur_x (no tail landmarks stored)

        Assumption: prompts in batch share the same length L.
        """
        self.eval()
        B, L, D = x.shape
        assert D == self.hidden_size
        max_seq_len = int(cache["max_seq_len"])
        assert L <= max_seq_len

        block_size = int(cache["block_size"])
        # IMPORTANT: require consistent block size
        if self.fixed_block_size is not None:
            assert block_size == self.fixed_block_size, "cache block_size != fixed_block_size"

        nb = int(math.ceil(L / block_size))
        Lpad = nb * block_size
        xpad = F.pad(x, (0, 0, 0, Lpad - L), value=0.0) if Lpad != L else x

        # token qkv (for full prompt)
        qkv = self.qkv_proj(xpad)
        q_tok, k_tok, v_tok = qkv.chunk(3, dim=-1)
        q_tok = self._split_heads(q_tok)  # [B,H,Lpad,Dh]
        k_tok = self._split_heads(k_tok)
        v_tok = self._split_heads(v_tok)

        if self.use_rope and self.rope_dim > 0:
            pos_tok = torch.arange(Lpad, device=x.device, dtype=torch.long)
            q_tok, k_tok = self._rope_apply_qk(q_tok, k_tok, pos_tok)

        # store token KV for real tokens only (first L)
        cache["tok_k"][:, :, :L, :] = k_tok[:, :, :L, :]
        cache["tok_v"][:, :, :L, :] = v_tok[:, :, :L, :]

        # local attention (causal), vectorized over full blocks
        lengths = torch.full((B,), L, device=x.device, dtype=torch.long)
        local_out = self._local_attention_vectorized(
            q_tok, k_tok, v_tok,
            lengths=lengths,
            block_size=block_size,
            is_causal=True,
            dropout_p=0.0,
        )[:, :, :L, :]  # [B,H,L,Dh]

        # ----- build landmarks for COMPLETE blocks only -----
        num_full_blocks = L // block_size
        tail_len = L - num_full_blocks * block_size

        cache["lm_len_sa"] = 0
        cache["lm_len"] = 0

        if num_full_blocks > 0:
            end_full = num_full_blocks * block_size
            x_full = x[:, :end_full, :].view(B, num_full_blocks, block_size, D)  # [B,nbf,bs,D]
            # all valid
            blk_valid = torch.ones((B, num_full_blocks, block_size), device=x.device, dtype=torch.bool)

            scores_pool = torch.einsum("md,bntd->bnmt", self.pool_queries, x_full) / math.sqrt(D)  # [B,nbf,M,bs]
            w_pool = masked_softmax(scores_pool, blk_valid[:, :, None, :], dim=-1, eps=self.eps).float()
            lm_raw = torch.einsum("bnmt,bntd->bnmd", w_pool, x_full.float()).to(dtype=x.dtype)    # [B,nbf,M,D]
            lm_raw = lm_raw.reshape(B, num_full_blocks * self.num_landmarks, D)                   # [B,Lm_full,D]
            Lm_full = lm_raw.shape[1]

            lm_valid = torch.ones((B, Lm_full), device=x.device, dtype=torch.bool)
            lm_block_idx = torch.arange(num_full_blocks, device=x.device).repeat_interleave(self.num_landmarks)  # [Lm_full]
            offsets = cache["lm_offsets"]
            lm_pos = (torch.arange(num_full_blocks, device=x.device, dtype=torch.long) * block_size).repeat_interleave(self.num_landmarks) \
                     + offsets.repeat(num_full_blocks)

            landmarks = lm_raw

            # landmark self-attn (block-level causal) + cache K/V (rope applied)
            if self.use_landmark_self_attn:
                lm_qkv = self.lm_qkv_proj(lm_raw)
                q_lm, k_lm, v_lm = lm_qkv.chunk(3, dim=-1)
                q_lm = self._split_heads(q_lm)
                k_lm = self._split_heads(k_lm)
                v_lm = self._split_heads(v_lm)
                if self.use_rope and self.rope_dim > 0:
                    q_lm, k_lm = self._rope_apply_qk(q_lm, k_lm, lm_pos)

                # store self-attn KV cache (for future landmarks)
                cache["lm_k_sa"][:, :, :Lm_full, :] = k_lm
                cache["lm_v_sa"][:, :, :Lm_full, :] = v_lm
                cache["lm_len_sa"] = Lm_full

                causal_lm = (lm_block_idx[None, :] > lm_block_idx[:, None])  # [Lm_full,Lm_full]

                def lm_mask_fn(s: int, e: int) -> Optional[torch.Tensor]:
                    return causal_lm[s:e, :]

                lm_out = attention_query_chunked(
                    q_lm, k_lm, v_lm,
                    key_valid=lm_valid,
                    mask_fn=lm_mask_fn,
                    chunk_size_q=1024,
                    attn_dropout=0.0,
                    training=False,
                    eps=self.eps,
                )
                lm_out = self._merge_heads(lm_out)
                lm_out = self.lm_out_proj(lm_out)
                landmarks = landmarks + lm_out

            # landmark -> token refinement (no dropout, streaming), only over tokens up to end_full
            if self.use_landmark_to_token:
                q_ref = self.lm_to_tok_q_proj(landmarks)
                q_ref = self._split_heads(q_ref)
                if self.use_rope and self.rope_dim > 0:
                    q_ref = self._rope_apply_q(q_ref, lm_pos)

                tok_k_full = k_tok[:, :, :end_full, :]
                tok_v_full = v_tok[:, :, :end_full, :]
                tok_block_idx = torch.arange(end_full, device=x.device) // block_size

                def lm_to_tok_mask_fn(ks: int, ke: int) -> Optional[torch.Tensor]:
                    kb = tok_block_idx[ks:ke]
                    return kb[None, :] > lm_block_idx[:, None]

                ref = attention_key_streaming(
                    q_ref, tok_k_full, tok_v_full,
                    key_valid=None,
                    mask_fn=lm_to_tok_mask_fn,
                    chunk_size_k=self.lm_to_token_k_chunk,
                    eps=self.eps,
                )
                ref = self._merge_heads(ref)
                ref = self.lm_to_tok_out_proj(ref)
                landmarks = landmarks + ref

            # token->landmark KV cache (rope on K)
            lm_kv = self.lm_kv_proj(landmarks)
            k_gl, v_gl = lm_kv.chunk(2, dim=-1)
            k_gl = self._split_heads(k_gl)
            v_gl = self._split_heads(v_gl)
            if self.use_rope and self.rope_dim > 0:
                k_gl = self._rope_apply_k(k_gl, lm_pos)

            cache["lm_k"][:, :, :Lm_full, :] = k_gl
            cache["lm_v"][:, :, :Lm_full, :] = v_gl
            cache["lm_len"] = Lm_full

            # ----- token -> landmark global for all prompt tokens -----
            q_tok_L = q_tok[:, :, :L, :]
            tok_block_idx_L = torch.arange(L, device=x.device) // block_size

            def tok_to_lm_mask_fn(s: int, e: int) -> Optional[torch.Tensor]:
                qb = tok_block_idx_L[s:e]
                return lm_block_idx[None, :] >= qb[:, None]

            global_out = attention_query_chunked(
                q_tok_L, k_gl, v_gl,
                key_valid=lm_valid,
                mask_fn=tok_to_lm_mask_fn,
                chunk_size_q=self.token_to_landmark_q_chunk,
                attn_dropout=0.0,
                training=False,
                eps=self.eps,
            )
        else:
            # no complete blocks -> global path is zero
            global_out = torch.zeros_like(local_out)

        # fuse + output projection
        if self.gating:
            gate = torch.sigmoid(self.gate_proj(x)).transpose(1, 2).unsqueeze(-1)  # [B,H,L,1]
            fused = local_out + gate * global_out
        else:
            fused = local_out + global_out

        y = self._merge_heads(fused)
        y = self.out_proj(y)

        # update tail state
        cache["t"] = L
        cache["cur_block_idx"] = num_full_blocks
        cache["cur_len"] = tail_len
        if tail_len > 0:
            cache["cur_x"][:, :tail_len, :] = x[:, num_full_blocks * block_size:, :]

        return y

    @torch.no_grad()
    def forward_step(self, x_t: torch.Tensor, cache: Dict[str, Any]) -> torch.Tensor:
        """
        One-token incremental step. Returns [B,1,D].
        """
        self.eval()
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        B, L1, D = x_t.shape
        assert L1 == 1 and D == self.hidden_size

        t = int(cache["t"])
        max_seq_len = int(cache["max_seq_len"])
        if t >= max_seq_len:
            raise RuntimeError("KV cache full.")

        block_size = int(cache["block_size"])
        cur_len = int(cache["cur_len"])
        cur_block_idx = int(cache["cur_block_idx"])

        # project qkv
        qkv = self.qkv_proj(x_t)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)  # [B,H,1,Dh]
        k = self._split_heads(k)
        v = self._split_heads(v)

        # RoPE at absolute position t
        if self.use_rope and self.rope_dim > 0:
            q = self._rope_apply_single(q, t)
            k = self._rope_apply_single(k, t)

        # store token KV
        cache["tok_k"][:, :, t:t+1, :] = k
        cache["tok_v"][:, :, t:t+1, :] = v

        # store x_t into current block buffer
        cache["cur_x"][:, cur_len:cur_len+1, :] = x_t
        cur_len += 1
        cache["cur_len"] = cur_len

        # local: attend within current block only (keys are all tokens in this block so far)
        block_start = cur_block_idx * block_size
        k_local = cache["tok_k"][:, :, block_start:t+1, :]
        v_local = cache["tok_v"][:, :, block_start:t+1, :]

        local_out = F.scaled_dot_product_attention(
            q, k_local, v_local,
            attn_mask=None, dropout_p=0.0, is_causal=False
        )  # [B,H,1,Dh]

        # global: attend to previous complete-block landmarks only (cache holds only previous blocks)
        lm_len = int(cache["lm_len"])
        if lm_len > 0:
            global_out = attention_query_chunked(
                q, cache["lm_k"][:, :, :lm_len, :], cache["lm_v"][:, :, :lm_len, :],
                key_valid=None,
                mask_fn=None,
                chunk_size_q=1,
                attn_dropout=0.0,
                training=False,
                eps=self.eps,
            )
        else:
            global_out = torch.zeros_like(local_out)

        if self.gating:
            gate = torch.sigmoid(self.gate_proj(x_t)).transpose(1, 2).unsqueeze(-1)  # [B,H,1,1]
            fused = local_out + gate * global_out
        else:
            fused = local_out + global_out

        y_t = self._merge_heads(fused)  # [B,1,D]
        y_t = self.out_proj(y_t)

        # advance time
        cache["t"] = t + 1

        # if block complete, append landmarks for this block for future steps
        if cur_len == block_size:
            end_pos = t + 1  # number of tokens available
            self._append_completed_block_landmarks(
                x_block=cache["cur_x"],
                tok_k=cache["tok_k"][:, :, :end_pos, :],
                tok_v=cache["tok_v"][:, :, :end_pos, :],
                block_idx=cur_block_idx,
                cache=cache,
            )
            cache["cur_len"] = 0
            cache["cur_block_idx"] = cur_block_idx + 1

        return y_t

    # --------------------
    # append a completed block's landmarks into cache (causal-safe)
    # --------------------
    @torch.no_grad()
    def _append_completed_block_landmarks(
        self,
        x_block: torch.Tensor,  # [B,bs,D]
        tok_k: torch.Tensor,    # [B,H,T,Dh] (T = end_pos of this block)
        tok_v: torch.Tensor,    # [B,H,T,Dh]
        block_idx: int,
        cache: Dict[str, Any],
    ) -> None:
        """
        Called only when a block is completed.
        After this, its landmarks become visible to future tokens.
        """
        B, bs, D = x_block.shape
        block_size = int(cache["block_size"])
        assert bs == block_size

        offsets = cache["lm_offsets"]                 # [M]
        lm_pos = (block_idx * block_size + offsets).to(device=x_block.device)  # [M]

        # pooling
        scores = torch.einsum("md,bsd->bms", self.pool_queries, x_block) / math.sqrt(D)  # [B,M,bs]
        valid = torch.ones((B, self.num_landmarks, bs), device=x_block.device, dtype=torch.bool)
        w = masked_softmax(scores, valid, dim=-1, eps=self.eps).float()
        lm = torch.einsum("bms,bsd->bmd", w, x_block.float()).to(dtype=x_block.dtype)  # [B,M,D]

        # landmark self-attn (new landmarks attend to all previous landmarks + themselves)
        if self.use_landmark_self_attn:
            lm_qkv = self.lm_qkv_proj(lm)
            q_lm, k_lm, v_lm = lm_qkv.chunk(3, dim=-1)
            q_lm = self._split_heads(q_lm)  # [B,H,M,Dh]
            k_lm = self._split_heads(k_lm)
            v_lm = self._split_heads(v_lm)

            if self.use_rope and self.rope_dim > 0:
                q_lm, k_lm = self._rope_apply_qk(q_lm, k_lm, lm_pos)

            lm_len_sa = int(cache["lm_len_sa"])
            # append self-attn KV cache
            cache["lm_k_sa"][:, :, lm_len_sa:lm_len_sa + self.num_landmarks, :] = k_lm
            cache["lm_v_sa"][:, :, lm_len_sa:lm_len_sa + self.num_landmarks, :] = v_lm

            # attend (no mask needed: previous blocks + same block only)
            k_all = cache["lm_k_sa"][:, :, :lm_len_sa + self.num_landmarks, :]
            v_all = cache["lm_v_sa"][:, :, :lm_len_sa + self.num_landmarks, :]

            out = attention_query_chunked(
                q_lm, k_all, v_all,
                key_valid=None,
                mask_fn=None,
                chunk_size_q=self.num_landmarks,
                attn_dropout=0.0,
                training=False,
                eps=self.eps,
            )
            out = self._merge_heads(out)
            out = self.lm_out_proj(out)
            lm = lm + out
            cache["lm_len_sa"] = lm_len_sa + self.num_landmarks

        # landmark -> token refinement (tokens only up to this block end_pos -> no mask needed)
        if self.use_landmark_to_token:
            q_ref = self.lm_to_tok_q_proj(lm)
            q_ref = self._split_heads(q_ref)
            if self.use_rope and self.rope_dim > 0:
                q_ref = self._rope_apply_q(q_ref, lm_pos)

            ref = attention_key_streaming(
                q_ref, tok_k, tok_v,
                key_valid=None,
                mask_fn=None,
                chunk_size_k=self.lm_to_token_k_chunk,
                eps=self.eps,
            )
            ref = self._merge_heads(ref)
            ref = self.lm_to_tok_out_proj(ref)
            lm = lm + ref

        # token->landmark KV cache
        lm_kv = self.lm_kv_proj(lm)
        k_gl, v_gl = lm_kv.chunk(2, dim=-1)
        k_gl = self._split_heads(k_gl)
        v_gl = self._split_heads(v_gl)
        if self.use_rope and self.rope_dim > 0:
            k_gl = self._rope_apply_k(k_gl, lm_pos)

        lm_len = int(cache["lm_len"])
        cache["lm_k"][:, :, lm_len:lm_len + self.num_landmarks, :] = k_gl
        cache["lm_v"][:, :, lm_len:lm_len + self.num_landmarks, :] = v_gl
        cache["lm_len"] = lm_len + self.num_landmarks


# ============================================================
# Transformer Block Wrapper (Pre-Norm)
# ============================================================

class SBLATransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_landmarks: int = 4,
        use_landmark_self_attn: bool = True,
        use_landmark_to_token: bool = True,
        use_rope: bool = True,
        fixed_block_size: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SBLAAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_landmarks=num_landmarks,
            attn_dropout=dropout,
            proj_dropout=dropout,
            use_rope=use_rope,
            use_landmark_self_attn=use_landmark_self_attn,
            use_landmark_to_token=use_landmark_to_token,
            gating=True,
            fixed_block_size=fixed_block_size,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, *, is_causal: bool = False) -> torch.Tensor:
        x_out, _, _ = self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + x_out
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# Quick self-test (CPU-friendly)
# ============================================================

def _quick_test():
    torch.manual_seed(0)
    B, L, D, H = 2, 257, 64, 4
    max_seq_len = 512
    block_size = int(math.ceil(math.sqrt(max_seq_len)))

    attn = SBLAAttention(
        hidden_size=D,
        num_heads=H,
        num_landmarks=3,
        use_rope=True,
        use_landmark_self_attn=True,
        use_landmark_to_token=True,
        gating=True,
        fixed_block_size=block_size,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ).eval()

    x = torch.randn(B, L, D)

    # full forward
    y_full, _, _ = attn(x)

    # cache prefill
    cache = attn.init_cache(batch_size=B, max_seq_len=max_seq_len, device=x.device, dtype=x.dtype)
    y_prefill = attn.prefill(x, cache)

    max_err = (y_full - y_prefill).abs().max().item()
    print("max |full - prefill| =", max_err)

    # step-by-step continuation consistency on next few tokens
    # build a longer sequence by appending some tokens
    extra = 10
    x2 = torch.randn(B, L + extra, D)
    x_prompt = x2[:, :L, :]
    x_next = x2[:, L:, :]

    cache2 = attn.init_cache(batch_size=B, max_seq_len=max_seq_len, device=x.device, dtype=x.dtype)
    y_prompt2 = attn.prefill(x_prompt, cache2)

    ys = [y_prompt2]
    for i in range(extra):
        y_t = attn.forward_step(x_next[:, i, :], cache2)  # [B,1,D]
        ys.append(y_t)

    y_inc = torch.cat(ys, dim=1)
    y_full2, _, _ = attn(x2)

    max_err2 = (y_inc - y_full2).abs().max().item()
    print("max |full2 - (prefill+steps)| =", max_err2)


if __name__ == "__main__":
    _quick_test()
