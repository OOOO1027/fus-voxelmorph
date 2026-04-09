# fUS-VoxelMorph GPU 设置指南

## 环境检查结果

### GPU 信息
- **GPU 型号**: NVIDIA GeForce RTX 5060 Laptop GPU
- **计算能力**: sm_120
- **驱动版本**: 572.97
- **CUDA 版本**: 12.8 (驱动), 12.4 (PyTorch)

### PyTorch 信息
- **PyTorch 版本**: 2.6.0+cu124
- **CUDA 可用**: True ⚠️ (有兼容性警告)
- **支持架构**: sm_50, sm_60, sm_61, sm_70, sm_75, sm_80, sm_86, sm_90

### 问题
PyTorch 2.6.0 不支持 RTX 5060 的 sm_120 架构，导致 GPU kernel 无法执行。

---

## 解决方案

### 方案 1: 使用 CPU 训练（当前可用）

```bash
python train_cpu.py --config configs/default.yaml --epochs 200
```

或带合成数据快速测试：
```bash
python train_cpu.py --synthetic --epochs 5 --batch_size 4
```

### 方案 2: 更新 PyTorch 到 Nightly 版本（推荐）

RTX 5060 需要 PyTorch 2.7+ 或 nightly 版本。安装命令：

```bash
# 安装 PyTorch nightly (支持 sm_120)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 或使用 pip upgrade
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

安装后验证：
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 方案 3: 使用 Conda 安装最新版本

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch-nightly -c nvidia
```

---

## 文件说明

### 训练脚本

| 文件 | 说明 | 使用条件 |
|------|------|----------|
| `train_cpu.py` | CPU 版本 | 所有环境 |
| `train_gpu_fp32.py` | GPU FP32 版本 | PyTorch 2.7+ 或 nightly |
| `train_gpu.py` | GPU + 混合精度 | PyTorch 2.7+ 或 nightly |
| `train.py` | 原文件 | 保持原样 |

### 模型文件

| 文件 | 说明 |
|------|------|
| `models/vxm2d.py` | GPU 优化版本（当前使用） |
| `models/vxm2d_cpu.py` | CPU 版本备份 |

---

## 更新 PyTorch 后测试

1. 更新 PyTorch 到 nightly 版本
2. 运行测试脚本：

```bash
# 测试 FP32 版本
python test_gpu_fp32_training.py

# 测试混合精度版本（如果 FP32 通过）
python test_gpu_training.py
```

3. 开始训练：

```bash
# FP32 版本
python train_gpu_fp32.py --synthetic --epochs 50

# 混合精度版本（更快）
python train_gpu.py --synthetic --epochs 50
```

---

## 性能预期

在 RTX 5060 Laptop GPU 上：

| 模式 | 预期速度 | 显存占用 |
|------|----------|----------|
| CPU | ~XX sec/epoch | N/A |
| GPU FP32 | ~5-10x faster | ~2-4 GB |
| GPU AMP | ~8-15x faster | ~1.5-3 GB |

---

## 代码修改总结

### train_gpu.py 主要修改
1. 使用 `torch.amp` 替代已弃用的 `torch.cuda.amp`
2. 添加 `autocast(device_type=device.type)`
3. 添加 `GradScaler(device=device.type)`
4. 使用 `non_blocking=True` 加速数据传输
5. 启用 `torch.backends.cudnn.benchmark`
6. 添加 `persistent_workers=True` 加速数据加载

### models/vxm2d.py 修改
1. 修复 SpatialTransformer grid 设备同步问题
2. 确保 grid 始终在正确的 device 上

---

## 注意事项

1. **显存管理**: 如遇到 OOM，减小 `batch_size` 或 `--num_workers`
2. **cuDNN**: GPU 版本自动启用 benchmark 模式
3. **混合精度**: 在支持的 GPU 上可加速 1.5-2 倍
4. **回退机制**: 如 CUDA 不可用，自动回退到 CPU

---

## 快速开始（更新 PyTorch 后）

```bash
# 1. 更新 PyTorch
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# 2. 测试
python test_gpu_fp32_training.py

# 3. 训练
python train_gpu_fp32.py --config configs/default.yaml
```
