#!/usr/bin/env python
"""
Quick test for GPU training (5 epochs)
Verify GPU + Mixed Precision training works correctly
"""

import torch
import time
import sys

def check_gpu():
    """Check GPU environment"""
    print("=" * 60)
    print("GPU Environment Check")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"Current device: {torch.cuda.current_device()}")
        
        # Test GPU computation
        print("\nTesting GPU computation...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"Matrix multiplication test (1000x1000): {elapsed*1000:.2f} ms")
        print("GPU computation: OK")
    else:
        print("WARNING: CUDA not available, will use CPU")
    
    print()

def test_amp():
    """Test mixed precision"""
    print("=" * 60)
    print("Mixed Precision (AMP) Test")
    print("=" * 60)
    
    try:
        from torch.amp import autocast, GradScaler
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create simple model
        model = torch.nn.Linear(100, 10).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = GradScaler(device=device.type)
        
        # Test data
        x = torch.randn(32, 100).to(device)
        target = torch.randn(32, 10).to(device)
        
        # Test AMP forward
        with autocast(device_type=device.type):
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)
        
        print(f"Loss dtype: {loss.dtype}")
        print(f"Output dtype: {output.dtype}")
        
        # Test AMP backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("Mixed precision training: OK")
        
    except Exception as e:
        print(f"Mixed precision test failed: {e}")
        return False
    
    print()
    return True

def test_training_loop():
    """Test complete training loop (5 epochs)"""
    print("=" * 60)
    print("5 Epochs Quick Training Test")
    print("=" * 60)
    
    # Import necessary modules
    from models import VxmDense2D
    from losses import NCC, Grad, RegistrationLoss
    from utils import get_gpu_memory
    from torch.amp import autocast, GradScaler
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = VxmDense2D(
        in_channels=1,
        enc_channels=[16, 32, 32, 32],
        dec_channels=[32, 32, 32, 32, 16, 16],
        integration_steps=7,
    ).to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create loss and optimizer
    sim_loss = NCC(win_size=9)
    reg_loss = Grad(penalty='l2')
    criterion = RegistrationLoss(sim_loss, reg_loss, reg_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler(device=device.type)
    
    # Simulate data
    batch_size = 4
    print(f"Batch size: {batch_size}")
    print()
    
    # Record GPU memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated() / (1024**2)
    
    # Train 5 epochs
    print("Starting training...")
    print("-" * 60)
    
    epoch_times = []
    
    for epoch in range(1, 6):
        model.train()
        epoch_start = time.time()
        
        # Simulate 10 batches per epoch
        for batch_idx in range(10):
            # Generate random data
            source = torch.randn(batch_size, 1, 128, 100).to(device)
            target = torch.randn(batch_size, 1, 128, 100).to(device)
            
            # Forward with AMP
            with autocast(device_type=device.type):
                warped, flow = model(source, target)
                loss, sim_loss_val, reg_loss_val = criterion(warped, target, flow)
            
            # Backward with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Print epoch info
        mem_str = ""
        if device.type == 'cuda':
            mem = get_gpu_memory()
            mem_str = f" | GPU: {mem['allocated']:.0f}MB"
        
        print(f"Epoch {epoch}/5 - Loss: {loss.item():.4f} | Time: {epoch_time:.2f}s{mem_str}")
    
    print("-" * 60)
    avg_time = sum(epoch_times) / len(epoch_times)
    print(f"Average epoch time: {avg_time:.2f}s")
    
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak GPU memory: {peak_mem:.1f} MB")
    
    print("\nTraining test: PASSED")
    print()
    
    return True

def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("fUS-VoxelMorph GPU Training Test")
    print("=" * 60)
    print()
    
    # Check GPU
    check_gpu()
    
    # Test AMP
    if not test_amp():
        print("AMP test failed, exiting")
        sys.exit(1)
    
    # Test training loop
    if not test_training_loop():
        print("Training test failed, exiting")
        sys.exit(1)
    
    print("=" * 60)
    print("All tests PASSED! GPU training is ready.")
    print("=" * 60)
    print()
    print("To start real training, run:")
    print("  python train_gpu.py --synthetic --epochs 50")

if __name__ == '__main__':
    main()
