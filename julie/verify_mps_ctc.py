import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"MPS Backend Available: {torch.backends.mps.is_available()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    
    # 1. Test basic MPS tensor allocation
    x = torch.ones(2, 2, device=device)
    print(f"Basic allocation check: {x}")
    
    # 2. Test CTC Loss on MPS (requires PR #176778)
    log_probs = torch.randn(50, 16, 20, device=device).log_softmax(2).requires_grad_()
    targets = torch.randint(1, 20, (16, 30), dtype=torch.long, device=device)
    input_lengths = torch.full((16,), 50, dtype=torch.long, device=device)
    target_lengths = torch.randint(10, 30, (16,), dtype=torch.long, device=device)
    
    try:
        loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        print("✅ Successfully computed CTC loss and backward pass on MPS!")
        print(f"Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Failed to run CTC loss on MPS: {e}")
else:
    print("❌ MPS is not available on this device/installation.")
