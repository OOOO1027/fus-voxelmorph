#!/usr/bin/env python
"""
验证 fUS-VoxelMorph GPU/CPU 设置
"""

import torch
import sys

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def check_pytorch():
    print_section("PyTorch 环境")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        capability = torch.cuda.get_device_capability(0)
        print(f"计算能力: sm_{capability[0]}{capability[1]}")
        
        # Check AMP support
        try:
            from torch.amp import autocast, GradScaler
            print("torch.amp: 可用")
        except ImportError:
            print("torch.amp: 不可用")
        
        # Test simple GPU operation
        try:
            x = torch.randn(100, 100).cuda()
            y = x @ x.T
            torch.cuda.synchronize()
            print("基础 GPU 运算: 正常")
        except Exception as e:
            print(f"基础 GPU 运算: 失败 - {e}")

def check_models():
    print_section("模型检查")
    try:
        from models import VxmDense2D
        model = VxmDense2D(integration_steps=7)
        params = sum(p.numel() for p in model.parameters())
        print(f"VxmDense2D 模型: OK")
        print(f"  参数量: {params:,}")
        
        # Test forward
        x = torch.randn(1, 1, 128, 100)
        warped, flow = model(x, x)
        print(f"  前向传播: OK (输出形状: {warped.shape}, {flow.shape})")
    except Exception as e:
        print(f"模型检查失败: {e}")

def check_losses():
    print_section("损失函数检查")
    try:
        from losses import NCC, Grad, RegistrationLoss
        
        sim_loss = NCC(win_size=9)
        reg_loss = Grad(penalty='l2')
        criterion = RegistrationLoss(sim_loss, reg_loss, reg_weight=1.0)
        
        warped = torch.randn(2, 1, 128, 100)
        target = torch.randn(2, 1, 128, 100)
        flow = torch.randn(2, 2, 128, 100)
        
        loss, sim, reg = criterion(warped, target, flow)
        print(f"损失函数: OK (loss={loss.item():.4f})")
    except Exception as e:
        print(f"损失函数检查失败: {e}")

def check_training_files():
    print_section("训练文件检查")
    import os
    
    files = {
        'train.py': '原始训练文件',
        'train_cpu.py': 'CPU 版本',
        'train_gpu.py': 'GPU + AMP 版本',
        'train_gpu_fp32.py': 'GPU FP32 版本',
        'test_gpu_training.py': 'GPU AMP 测试',
        'test_gpu_fp32_training.py': 'GPU FP32 测试',
    }
    
    for file, desc in files.items():
        status = "[OK]" if os.path.exists(file) else [MISSING]
        print(f"{status} {file:<30} {desc}")

def recommend_mode():
    print_section("推荐使用模式")
    
    if not torch.cuda.is_available():
        print("CUDA 不可用，推荐使用 CPU 模式:")
        print("  python train_cpu.py --synthetic --epochs 50")
        return
    
    capability = torch.cuda.get_device_capability(0)
    sm = capability[0] * 10 + capability[1]
    
    if sm > 90:
        print(f"检测到 sm_{sm} 架构，需要 PyTorch 2.7+ 或 nightly 版本")
        print("\n当前 PyTorch 版本可能不完全支持您的 GPU。")
        print("选项 1: 使用 CPU 模式（当前可用）")
        print("  python train_cpu.py --synthetic --epochs 50")
        print("\n选项 2: 更新 PyTorch 后使用 GPU 模式")
        print("  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128")
        print("  python train_gpu_fp32.py --synthetic --epochs 50")
    else:
        print("GPU 完全支持，推荐使用 GPU 模式:")
        print("  python train_gpu.py --synthetic --epochs 50")

def main():
    print("\n" + "=" * 60)
    print("fUS-VoxelMorph 设置验证")
    print("=" * 60)
    
    check_pytorch()
    check_models()
    check_losses()
    check_training_files()
    recommend_mode()
    
    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)

if __name__ == '__main__':
    main()
