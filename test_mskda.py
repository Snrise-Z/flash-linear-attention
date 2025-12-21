#!/usr/bin/env python
"""
Test script for Multi-Scale KDA (MSKDA) implementation.
Verifies that the model can be instantiated and run forward/backward passes.
"""
import torch
import torch.nn as nn

def test_mskda_layer():
    """Test MultiScaleKimiDeltaAttention layer."""
    print("=" * 60)
    print("Testing MultiScaleKimiDeltaAttention layer...")
    print("=" * 60)
    
    from fla.layers.mskda import MultiScaleKimiDeltaAttention, compute_log_spaced_timescales
    
    # Test log-spaced timescales computation
    print("\n1. Testing compute_log_spaced_timescales...")
    tau_min, tau_max = 1.0, 1000.0
    
    # Test with different group sizes
    for num_groups in [1, 4, 16, 64]:
        A_log = compute_log_spaced_timescales(num_groups, tau_min, tau_max)
        print(f"   num_groups={num_groups}: A_log shape={A_log.shape}")
        if num_groups > 1:
            # Verify log-spacing
            taus = 1.0 / A_log.exp()
            print(f"   Time constants: min={taus.min():.2f}, max={taus.max():.2f}")
    print("   ✓ Log-spaced timescales computation passed")
    
    # Test layer instantiation with different configurations
    print("\n2. Testing layer instantiation...")
    
    configs = [
        {"num_a_groups": 1, "name": "per-head A (original KDA behavior)"},
        {"num_a_groups": 4, "name": "grouped A (4 groups)"},
        {"num_a_groups": 64, "name": "per-channel A (default)"},
    ]
    
    for cfg in configs:
        layer = MultiScaleKimiDeltaAttention(
            hidden_size=512,
            head_dim=64,
            num_heads=8,
            num_a_groups=cfg["num_a_groups"],
            tau_min=1.0,
            tau_max=1000.0,
            use_short_conv=False,
            allow_neg_eigval=False,
            use_tanh_beta=False,
            mode="chunk",
        )
        print(f"   {cfg['name']}: A_log shape={layer.A_log.shape}")
    print("   ✓ Layer instantiation passed")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    layer = MultiScaleKimiDeltaAttention(
        hidden_size=512,
        head_dim=64,
        num_heads=8,
        num_a_groups=64,  # per-channel
        tau_min=1.0,
        tau_max=1000.0,
        use_short_conv=False,
        allow_neg_eigval=False,
        use_tanh_beta=False,
        mode="chunk",
    ).to(device, dtype=dtype)
    
    batch_size, seq_len = 2, 128
    x = torch.randn(batch_size, seq_len, 512, device=device, dtype=dtype)
    
    with torch.no_grad():
        out, _, _ = layer(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"
    print("   ✓ Forward pass passed")
    
    # Test backward pass
    print("\n4. Testing backward pass...")
    layer.train()
    x = torch.randn(batch_size, seq_len, 512, device=device, dtype=dtype, requires_grad=True)
    out, _, _ = layer(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Input gradient is None"
    assert layer.A_log.grad is not None, "A_log gradient is None"
    print(f"   A_log grad: min={layer.A_log.grad.min():.6f}, max={layer.A_log.grad.max():.6f}")
    print("   ✓ Backward pass passed")
    
    # Test with tanh beta
    print("\n5. Testing with use_tanh_beta=True...")
    layer_tanh = MultiScaleKimiDeltaAttention(
        hidden_size=512,
        head_dim=64,
        num_heads=8,
        num_a_groups=64,
        tau_min=1.0,
        tau_max=1000.0,
        use_short_conv=False,
        allow_neg_eigval=False,
        use_tanh_beta=True,  # Use 1+tanh parameterization
        mode="chunk",
    ).to(device, dtype=dtype)
    
    x = torch.randn(batch_size, seq_len, 512, device=device, dtype=dtype)
    with torch.no_grad():
        out_tanh, _, _ = layer_tanh(x)
    print(f"   Output shape with tanh beta: {out_tanh.shape}")
    print("   ✓ Tanh beta test passed")
    
    print("\n" + "=" * 60)
    print("MultiScaleKimiDeltaAttention layer tests PASSED!")
    print("=" * 60)


def test_mskda_model():
    """Test MSKDA model."""
    print("\n" + "=" * 60)
    print("Testing MSKDA model...")
    print("=" * 60)
    
    from transformers import AutoModelForCausalLM
    from fla.models import MSKDAConfig
    
    # Test config
    print("\n1. Testing MSKDAConfig...")
    config = MSKDAConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_heads=4,
        head_dim=64,
        num_a_groups=64,  # per-channel
        tau_min=1.0,
        tau_max=1000.0,
        use_tanh_beta=False,
        vocab_size=1000,
        max_position_embeddings=512,
        fuse_norm=True,
        fuse_swiglu=True,
        fuse_cross_entropy=True,
    )
    print(f"   num_a_groups: {config.num_a_groups}")
    print(f"   tau_min: {config.tau_min}, tau_max: {config.tau_max}")
    print("   ✓ Config creation passed")
    
    # Test model instantiation
    print("\n2. Testing model instantiation...")
    model = AutoModelForCausalLM.from_config(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model type: {type(model).__name__}")
    print(f"   Number of parameters: {num_params:,}")
    print("   ✓ Model instantiation passed")
    
    # Test forward pass
    print("\n3. Testing model forward pass...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    model = model.to(device, dtype=dtype)
    
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
    
    print(f"   Loss: {outputs.loss.item():.4f}")
    # Note: logits may be None when using fuse_cross_entropy=True (which is the default)
    if outputs.logits is not None:
        print(f"   Logits shape: {outputs.logits.shape}")
    else:
        print("   Logits: None (fused cross-entropy mode)")
    print("   ✓ Forward pass passed")
    
    # Test backward pass
    print("\n4. Testing model backward pass...")
    model.train()
    outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
    outputs.loss.backward()
    
    # Check gradients
    for name, param in model.named_parameters():
        if "A_log" in name and param.grad is not None:
            print(f"   {name}: grad min={param.grad.min():.6f}, max={param.grad.max():.6f}")
    print("   ✓ Backward pass passed")
    
    # Test different configurations
    print("\n5. Testing different num_a_groups configurations...")
    for num_a_groups in [1, 4, 64]:
        cfg = MSKDAConfig(
            hidden_size=256,
            num_hidden_layers=2,
            num_heads=4,
            head_dim=64,
            num_a_groups=num_a_groups,
            vocab_size=1000,
            max_position_embeddings=512,
        )
        m = AutoModelForCausalLM.from_config(cfg)
        m = m.to(device, dtype=dtype)
        
        with torch.no_grad():
            out = m(input_ids=input_ids, use_cache=False)
        
        print(f"   num_a_groups={num_a_groups}: OK")
    print("   ✓ Different configurations passed")
    
    print("\n" + "=" * 60)
    print("MSKDA model tests PASSED!")
    print("=" * 60)


def test_multiscale_gate():
    """Test multi-scale gate computation."""
    print("\n" + "=" * 60)
    print("Testing multi-scale gate computation...")
    print("=" * 60)
    
    from fla.layers.mskda import multiscale_gate_forward, compute_log_spaced_timescales
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    batch_size, seq_len, H, K = 2, 128, 8, 64
    
    # Test per-head A (original KDA behavior)
    print("\n1. Testing per-head A (num_groups=1)...")
    g = torch.randn(batch_size, seq_len, H, K, device=device, dtype=dtype)
    A_log = compute_log_spaced_timescales(1, tau_min=1.0, tau_max=1000.0).unsqueeze(0).expand(H, -1).to(device)
    dt_bias = torch.randn(H * K, device=device, dtype=dtype)
    
    g_out = multiscale_gate_forward(g, A_log, dt_bias, num_groups=1)
    print(f"   A_log shape: {A_log.shape}")
    print(f"   Output shape: {g_out.shape}")
    print(f"   Output range: [{g_out.min():.4f}, {g_out.max():.4f}]")
    assert g_out.shape == g.shape
    print("   ✓ Per-head A passed")
    
    # Test grouped A
    print("\n2. Testing grouped A (num_groups=4)...")
    A_log = compute_log_spaced_timescales(4, tau_min=1.0, tau_max=1000.0).unsqueeze(0).expand(H, -1).to(device)
    
    g_out = multiscale_gate_forward(g, A_log, dt_bias, num_groups=4)
    print(f"   A_log shape: {A_log.shape}")
    print(f"   Output shape: {g_out.shape}")
    print(f"   Output range: [{g_out.min():.4f}, {g_out.max():.4f}]")
    assert g_out.shape == g.shape
    print("   ✓ Grouped A passed")
    
    # Test per-channel A
    print("\n3. Testing per-channel A (num_groups=K)...")
    A_log = compute_log_spaced_timescales(K, tau_min=1.0, tau_max=1000.0).unsqueeze(0).expand(H, -1).to(device)
    
    g_out = multiscale_gate_forward(g, A_log, dt_bias, num_groups=K)
    print(f"   A_log shape: {A_log.shape}")
    print(f"   Output shape: {g_out.shape}")
    print(f"   Output range: [{g_out.min():.4f}, {g_out.max():.4f}]")
    assert g_out.shape == g.shape
    print("   ✓ Per-channel A passed")
    
    # Verify multi-scale property: different channels should have different decay rates
    print("\n4. Verifying multi-scale property...")
    # Check that first and last group have different A values
    A_first = A_log[0, 0].item()
    A_last = A_log[0, -1].item()
    print(f"   A_log[0, 0] = {A_first:.4f} (rate = {math.exp(A_first):.4f})")
    print(f"   A_log[0, -1] = {A_last:.4f} (rate = {math.exp(A_last):.4f})")
    assert abs(A_first - A_last) > 0.1, "A values should be different across groups"
    print("   ✓ Multi-scale property verified")
    
    print("\n" + "=" * 60)
    print("Multi-scale gate tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    import math
    
    print("=" * 60)
    print("MSKDA (Multi-Scale KDA) Implementation Tests")
    print("=" * 60)
    
    test_multiscale_gate()
    test_mskda_layer()
    test_mskda_model()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nMSKDA implementation is ready for use.")
    print("\nTo train MSKDA on WikiText-103:")
    print("  python examples/train_mskda_wikitext103_epochs20.py")
    print("\nKey arguments:")
    print("  --num_a_groups N   # Number of groups for A (default: head_dim for per-channel)")
    print("  --tau_min 1.0      # Min time constant (fastest decay)")
    print("  --tau_max 1000.0   # Max time constant (slowest decay)")
    print("  --use_tanh_beta    # Use 1+tanh parameterization for beta")
