#!/usr/bin/env python3
"""
Test SBLA KV-cache incremental decoding functionality.
Validates that prefill + forward_step produce identical results to full forward.
"""

import torch
import math
from fla.layers import SBLAAttention

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_ROPE = torch.cuda.is_available()  # RoPE requires CUDA (Triton kernels)

print(f"Running KV-cache tests on device: {DEVICE}")
print(f"RoPE enabled: {USE_ROPE}")
print()


def test_prefill_consistency():
    """Test that full forward and prefill produce the same outputs"""
    print("=" * 60)
    print("Test 1: Prefill Consistency")
    print("=" * 60)
    
    torch.manual_seed(0)
    B, L, D, H = 2, 257, 64, 4
    max_seq_len = 512
    block_size = int(math.ceil(math.sqrt(max_seq_len)))

    attn = SBLAAttention(
        hidden_size=D,
        num_heads=H,
        num_landmarks=3,
        use_rope=USE_ROPE,
        use_landmark_self_attn=True,
        use_landmark_to_token=True,
        gating=True,
        fixed_block_size=block_size,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ).to(DEVICE).eval()

    x = torch.randn(B, L, D, device=DEVICE)

    # Full forward
    y_full, _, _ = attn(x, attention_mask=None)

    # Cache prefill
    cache = attn.init_cache(batch_size=B, max_seq_len=max_seq_len, device=DEVICE, dtype=x.dtype)
    y_prefill = attn.prefill(x, cache)

    max_err = (y_full - y_prefill).abs().max().item()
    mean_err = (y_full - y_prefill).abs().mean().item()
    
    print(f"Max |full - prefill| = {max_err:.2e}")
    print(f"Mean |full - prefill| = {mean_err:.2e}")
    
    # Check numerical consistency
    tolerance = 1e-4 if USE_ROPE else 1e-5
    assert max_err < tolerance, f"Max error {max_err} exceeds tolerance {tolerance}"
    
    print("✓ Prefill consistency test PASSED")
    print()
    return True


def test_incremental_decoding():
    """Test that prefill + step-by-step decoding matches full forward"""
    print("=" * 60)
    print("Test 2: Incremental Decoding Consistency")
    print("=" * 60)
    
    torch.manual_seed(0)
    B, L, D, H = 2, 257, 64, 4
    max_seq_len = 512
    block_size = int(math.ceil(math.sqrt(max_seq_len)))
    extra = 10  # Additional tokens to decode

    attn = SBLAAttention(
        hidden_size=D,
        num_heads=H,
        num_landmarks=3,
        use_rope=USE_ROPE,
        use_landmark_self_attn=True,
        use_landmark_to_token=True,
        gating=True,
        fixed_block_size=block_size,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ).to(DEVICE).eval()

    # Build a longer sequence
    x_full = torch.randn(B, L + extra, D, device=DEVICE)
    x_prompt = x_full[:, :L, :]
    x_next = x_full[:, L:, :]

    # Method 1: Full forward on entire sequence
    y_full, _, _ = attn(x_full, attention_mask=None)

    # Method 2: Prefill + incremental steps
    cache = attn.init_cache(batch_size=B, max_seq_len=max_seq_len, device=DEVICE, dtype=x_full.dtype)
    y_prefill = attn.prefill(x_prompt, cache)
    
    ys = [y_prefill]
    for i in range(extra):
        y_t = attn.forward_step(x_next[:, i, :], cache)  # [B,1,D]
        ys.append(y_t)

    y_incremental = torch.cat(ys, dim=1)

    max_err = (y_incremental - y_full).abs().max().item()
    mean_err = (y_incremental - y_full).abs().mean().item()
    
    print(f"Max |full - (prefill+steps)| = {max_err:.2e}")
    print(f"Mean |full - (prefill+steps)| = {mean_err:.2e}")
    
    # Check numerical consistency
    tolerance = 2e-4 if USE_ROPE else 1e-5
    assert max_err < tolerance, f"Max error {max_err} exceeds tolerance {tolerance}"
    
    print("✓ Incremental decoding consistency test PASSED")
    print()
    return True


def test_cache_block_completion():
    """Test that cache correctly handles block completion"""
    print("=" * 60)
    print("Test 3: Cache Block Completion")
    print("=" * 60)
    
    torch.manual_seed(42)
    B, D, H = 1, 64, 4
    block_size = 8
    max_seq_len = 64

    attn = SBLAAttention(
        hidden_size=D,
        num_heads=H,
        num_landmarks=2,
        use_rope=USE_ROPE,
        use_landmark_self_attn=True,
        use_landmark_to_token=True,
        gating=True,
        fixed_block_size=block_size,
    ).to(DEVICE).eval()

    cache = attn.init_cache(batch_size=B, max_seq_len=max_seq_len, device=DEVICE, dtype=torch.float32)
    
    # Generate tokens one by one
    for t in range(block_size * 2):  # Generate 2 full blocks
        x_t = torch.randn(B, 1, D, device=DEVICE)
        y_t = attn.forward_step(x_t, cache)
        
        # Check cache state
        assert cache["t"] == t + 1, f"Cache time should be {t+1}, got {cache['t']}"
        
        # Check landmark cache after each block completion
        expected_lm_len = (t + 1) // block_size * attn.num_landmarks
        assert cache["lm_len"] == expected_lm_len, \
            f"At t={t+1}, expected {expected_lm_len} landmarks, got {cache['lm_len']}"
        
        # Check output shape
        assert y_t.shape == (B, 1, D), f"Output shape should be (B,1,D), got {y_t.shape}"
    
    print(f"✓ Generated {block_size * 2} tokens (2 blocks)")
    print(f"✓ Landmark cache contains {cache['lm_len']} landmarks ({attn.num_landmarks} per block)")
    print(f"✓ Cache block completion test PASSED")
    print()
    return True


def test_cache_overflow_protection():
    """Test that cache correctly rejects tokens beyond max_seq_len"""
    print("=" * 60)
    print("Test 4: Cache Overflow Protection")
    print("=" * 60)
    
    torch.manual_seed(123)
    B, D, H = 1, 32, 2
    max_seq_len = 16

    attn = SBLAAttention(
        hidden_size=D,
        num_heads=H,
        num_landmarks=2,
        use_rope=USE_ROPE,
        fixed_block_size=4,
    ).to(DEVICE).eval()

    cache = attn.init_cache(batch_size=B, max_seq_len=max_seq_len, device=DEVICE, dtype=torch.float32)
    
    # Fill up to max_seq_len
    for t in range(max_seq_len):
        x_t = torch.randn(B, 1, D, device=DEVICE)
        y_t = attn.forward_step(x_t, cache)
    
    print(f"✓ Successfully generated {max_seq_len} tokens")
    
    # Try to exceed max_seq_len
    try:
        x_overflow = torch.randn(B, 1, D, device=DEVICE)
        y_overflow = attn.forward_step(x_overflow, cache)
        assert False, "Should have raised RuntimeError for cache overflow"
    except RuntimeError as e:
        assert "KV cache full" in str(e), f"Expected 'KV cache full' error, got: {e}"
        print(f"✓ Correctly raised RuntimeError on overflow: {e}")
    
    print("✓ Cache overflow protection test PASSED")
    print()
    return True


def test_varying_batch_positions():
    """Test cache with different prompt lengths (simulating batched prefill)"""
    print("=" * 60)
    print("Test 5: Varying Prompt Lengths")
    print("=" * 60)
    
    torch.manual_seed(456)
    B, D, H = 2, 64, 4
    max_seq_len = 128
    block_size = 16
    
    # Different prompt lengths for each batch item
    L1, L2 = 33, 49

    attn = SBLAAttention(
        hidden_size=D,
        num_heads=H,
        num_landmarks=3,
        use_rope=USE_ROPE,
        fixed_block_size=block_size,
    ).to(DEVICE).eval()

    # Note: Current implementation assumes same length across batch
    # We test with same-length prompts but different masks
    L = max(L1, L2)
    x = torch.randn(B, L, D, device=DEVICE)
    
    # Create attention mask (True = padding)
    attention_mask = torch.zeros(B, L, dtype=torch.bool, device=DEVICE)
    attention_mask[0, L1:] = True  # Pad sequence 1 after L1
    attention_mask[1, L2:] = True  # Pad sequence 2 after L2
    
    # Full forward with mask
    y_full, _, _ = attn(x, attention_mask=attention_mask)
    
    # Check that padded positions are zero
    assert torch.allclose(y_full[0, L1:], torch.zeros_like(y_full[0, L1:])), \
        "Padded positions should be zero (sequence 1)"
    assert torch.allclose(y_full[1, L2:], torch.zeros_like(y_full[1, L2:])), \
        "Padded positions should be zero (sequence 2)"
    
    print(f"✓ Batch item 1: {L1} valid tokens, {L-L1} padding")
    print(f"✓ Batch item 2: {L2} valid tokens, {L-L2} padding")
    print("✓ Padded positions correctly zeroed")
    print("✓ Varying prompt lengths test PASSED")
    print()
    return True


def main():
    print("=" * 60)
    print("SBLA KV-Cache Incremental Decoding Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        ("Prefill Consistency", test_prefill_consistency),
        ("Incremental Decoding", test_incremental_decoding),
        ("Cache Block Completion", test_cache_block_completion),
        ("Cache Overflow Protection", test_cache_overflow_protection),
        ("Varying Prompt Lengths", test_varying_batch_positions),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED with error: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"KV-Cache Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All KV-cache tests PASSED!")
        print("\nSBLA incremental decoding is working correctly:")
        print("  ✓ init_cache() initializes buffers")
        print("  ✓ prefill() matches full forward")
        print("  ✓ forward_step() maintains consistency")
        print("  ✓ _append_completed_block_landmarks() updates cache correctly")
        print("  ✓ Overflow protection works")
        return 0
    else:
        print(f"\n❌ {failed} KV-cache test(s) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
