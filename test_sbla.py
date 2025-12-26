#!/usr/bin/env python3
# Test script for SBLA (Sqrt-Block Landmark Attention)

import torch
import torch.nn as nn
from fla.layers import SBLAAttention

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_ROPE = torch.cuda.is_available()  # Only use RoPE with CUDA (Triton kernels require GPU)

print(f"Running tests on device: {DEVICE}")
print(f"RoPE enabled: {USE_ROPE}")
print()

def test_sbla_basic():
    """Test basic forward pass of SBLA"""
    print("Testing SBLA basic forward pass...")
    
    # Config
    B, L, D = 2, 128, 256
    num_heads = 8
    num_landmarks = 4
    block_size = 16  # Fixed block size for consistency
    
    # Create layer
    sbla = SBLAAttention(
        hidden_size=D,
        num_heads=num_heads,
        num_landmarks=num_landmarks,
        use_rope=USE_ROPE,
        rope_theta=10000.,
        use_landmark_self_attn=True,
        use_landmark_to_token=True,
        gating=True,
        fixed_block_size=block_size,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ).to(DEVICE)
    
    # Random input
    x = torch.randn(B, L, D, device=DEVICE)
    
    # Forward pass
    with torch.no_grad():
        sbla.eval()
        output, attn_weights, cache = sbla(x)
    
    # Check output shape
    assert output.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {output.shape}"
    print(f"✓ Output shape correct: {output.shape}")
    
    # Check that output is not NaN
    assert not torch.isnan(output).any(), "Output contains NaN values"
    print("✓ No NaN values in output")
    
    # Check that output is not all zeros
    assert output.abs().sum() > 0, "Output is all zeros"
    print("✓ Output is not all zeros")
    
    print("✓ Basic forward pass test PASSED\n")
    return True


def test_sbla_with_padding():
    """Test SBLA with padding mask"""
    print("Testing SBLA with padding mask...")
    
    # Config
    B, L, D = 2, 100, 128
    num_heads = 4
    
    # Create layer
    sbla = SBLAAttention(
        hidden_size=D,
        num_heads=num_heads,
        num_landmarks=3,
        fixed_block_size=10,
        use_rope=USE_ROPE,
    ).to(DEVICE)
    
    # Random input
    x = torch.randn(B, L, D, device=DEVICE)
    
    # Create padding mask (True = padding)
    # Sequence 1: 80 valid tokens, 20 padding
    # Sequence 2: 60 valid tokens, 40 padding
    attention_mask = torch.zeros(B, L, dtype=torch.bool, device=DEVICE)
    attention_mask[0, 80:] = True
    attention_mask[1, 60:] = True
    
    # Forward pass
    with torch.no_grad():
        sbla.eval()
        output, _, _ = sbla(x, attention_mask=attention_mask)
    
    # Check output shape
    assert output.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {output.shape}"
    print(f"✓ Output shape correct: {output.shape}")
    
    # Check that padded positions are zero
    assert torch.allclose(output[0, 80:], torch.zeros_like(output[0, 80:])), \
        "Padded positions should be zero (sequence 1)"
    assert torch.allclose(output[1, 60:], torch.zeros_like(output[1, 60:])), \
        "Padded positions should be zero (sequence 2)"
    print("✓ Padded positions are correctly zeroed")
    
    # Check that valid positions are non-zero
    assert output[0, :80].abs().sum() > 0, "Valid positions should be non-zero (sequence 1)"
    assert output[1, :60].abs().sum() > 0, "Valid positions should be non-zero (sequence 2)"
    print("✓ Valid positions are non-zero")
    
    print("✓ Padding mask test PASSED\n")
    return True


def test_sbla_different_configs():
    """Test SBLA with different configurations"""
    print("Testing SBLA with different configurations...")
    
    configs = [
        {"use_rope": False, "use_landmark_self_attn": False, "use_landmark_to_token": False, "gating": False},
        {"use_rope": USE_ROPE, "use_landmark_self_attn": True, "use_landmark_to_token": False, "gating": True},
        {"use_rope": USE_ROPE, "use_landmark_self_attn": False, "use_landmark_to_token": True, "gating": False},
        {"use_rope": USE_ROPE, "use_landmark_self_attn": True, "use_landmark_to_token": True, "gating": True},
    ]
    
    B, L, D = 1, 64, 128
    num_heads = 4
    x = torch.randn(B, L, D, device=DEVICE)
    
    for i, config in enumerate(configs):
        print(f"  Config {i+1}: {config}")
        sbla = SBLAAttention(
            hidden_size=D,
            num_heads=num_heads,
            num_landmarks=2,
            fixed_block_size=8,
            **config
        ).to(DEVICE)
        
        with torch.no_grad():
            sbla.eval()
            output, _, _ = sbla(x)
        
        assert output.shape == (B, L, D), f"Config {i+1} failed: wrong shape"
        assert not torch.isnan(output).any(), f"Config {i+1} failed: NaN values"
        assert output.abs().sum() > 0, f"Config {i+1} failed: all zeros"
        print(f"    ✓ Config {i+1} passed")
    
    print("✓ Different configurations test PASSED\n")
    return True


def test_sbla_gradient_flow():
    """Test that gradients flow correctly through SBLA"""
    print("Testing gradient flow through SBLA...")
    
    B, L, D = 2, 64, 128
    num_heads = 4
    
    # Create layer
    sbla = SBLAAttention(
        hidden_size=D,
        num_heads=num_heads,
        num_landmarks=3,
        fixed_block_size=8,
        use_rope=USE_ROPE,
    ).to(DEVICE)
    sbla.train()
    
    # Random input with gradients
    x = torch.randn(B, L, D, requires_grad=True, device=DEVICE)
    
    # Forward pass
    output, _, _ = sbla(x)
    
    # Compute loss and backward
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
    print("✓ Input gradients computed correctly")
    
    # Check that all module parameters have gradients
    for name, param in sbla.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient contains NaN for: {name}"
    print("✓ All parameter gradients computed correctly")
    
    print("✓ Gradient flow test PASSED\n")
    return True


def main():
    print("=" * 60)
    print("SBLA (Sqrt-Block Landmark Attention) Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        ("Basic Forward Pass", test_sbla_basic),
        ("Padding Mask", test_sbla_with_padding),
        ("Different Configurations", test_sbla_different_configs),
        ("Gradient Flow", test_sbla_gradient_flow),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_fn in tests:
        try:
            print(f"Running: {test_name}")
            print("-" * 60)
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED with error: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
