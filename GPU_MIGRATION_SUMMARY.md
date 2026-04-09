# fUS-VoxelMorph GPU 迁移总结

## 完成的工作

### 1. 环境检查 ✅
- GPU 型号: NVIDIA GeForce RTX 5060 Laptop GPU (sm_120)
- 驱动版本: 572.97
- CUDA 版本: 12.4 (PyTorch), 12.8 (驱动)
- PyTorch 版本: 2.6.0+cu124

**发现**: RTX 5060 (sm_120) 需要 PyTorch 2.7+ 或 nightly 版本支持。

### 2. 文件备份 ✅

| 原文件 | 备份文件 | 说明 |
|--------|----------|------|
| `train.py` | `train_cpu.py` | CPU 版本备份 |
| `train.py` | `train_gpu.py` | GPU + AMP 版本 |
| `train.py` | `train_gpu_fp32.py` | GPU FP32 版本 |
| `models/vxm2d.py` | `models/vxm2d_cpu.py` | 模型备份 |
| `losses/losses.py` | `losses/losses_cpu.py` | 损失函数备份 |

### 3. 代码修改 ✅

#### train_gpu.py 修改内容：

1. **导入更新**:
   ```python
   from torch.amp import autocast, GradScaler  # 新 API
   ```

2. **Device 设置**:
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   torch.backends.cudnn.benchmark = True  # 启用 cuDNN benchmark
   ```

3. **混合精度训练**:
   ```python
   with autocast(device_type=device.type):
       warped, flow = model(source, target)
       loss, sim_loss, reg_loss = criterion(warped, target, flow)
   
   scaler = GradScaler(device=device.type)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

4. **数据加载优化**:
   ```python
   source = source.to(device, non_blocking=True)  # 非阻塞传输
   pin_memory=True  # 固定内存
   persistent_workers=True  # 持久化工作进程
   ```

5. **GPU 监控**:
   ```python
   # 每个 epoch 打印显存占用
   gpu_mem = get_gpu_memory()
   console_logger.info(f"GPU: {gpu_mem['allocated']:.0f}MB")
   
   # TensorBoard 记录更多显存指标
   tb_logger.log_scalar('gpu/memory_reserved_mb', gpu_mem['reserved'], epoch)
   tb_logger.log_scalar('gpu/max_allocated_mb', gpu_mem['max_allocated'], epoch)
   ```

#### train_gpu_fp32.py 修改内容：
- 与 train_gpu.py 相同，但移除了 AMP 相关代码
- 使用纯 FP32 训练
- 适用于不支持混合精度的 GPU

#### models/vxm2d.py 修改内容：

```python
def _ensure_grid(self, displacement):
    size = tuple(displacement.shape[2:])
    if size != self._cur_size:
        self._build_grid(size)
    # 确保 grid 始终在正确的 device 上
    if self.grid.device != displacement.device:
        self.grid = self.grid.to(displacement.device)
```

### 4. 统一 Device 设置 ✅

所有文件使用统一的 device 设置：
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

检查过以下文件：
- ✅ `train_gpu.py`
- ✅ `train_gpu_fp32.py`
- ✅ `evaluate.py`
- ✅ `register.py`
- ✅ `compare_baseline.py`
- ✅ `train_v2.py`
- ✅ `scripts/demo_comparison.py`
- ✅ `scripts/demo_visualization.py`

### 5. 测试脚本 ✅

| 测试脚本 | 说明 |
|----------|------|
| `test_gpu_training.py` | 测试 GPU + AMP |
| `test_gpu_fp32_training.py` | 测试 GPU FP32 |

---

## 当前状态

### 可用 ✅
- CPU 训练: `python train_cpu.py --synthetic --epochs 5`

### 需要更新 PyTorch ⚠️
- GPU FP32 训练: `python train_gpu_fp32_training.py`
- GPU AMP 训练: `python test_gpu_training.py`

**需要**: 安装 PyTorch nightly 版本以支持 RTX 5060 (sm_120)

---

## 下一步操作

### 1. 更新 PyTorch (推荐)

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 2. 验证安装

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python test_gpu_fp32_training.py
```

### 3. 开始训练

```bash
# FP32 版本
python train_gpu_fp32.py --synthetic --epochs 50 --batch_size 8

# 或混合精度版本（更快）
python train_gpu.py --synthetic --epochs 50 --batch_size 8
```

---

## 文件清单

### 训练脚本
- `train.py` - 原始文件（保持不变）
- `train_cpu.py` - CPU 版本
- `train_gpu.py` - GPU + AMP 版本
- `train_gpu_fp32.py` - GPU FP32 版本

### 模型文件
- `models/vxm2d.py` - GPU 优化版本
- `models/vxm2d_cpu.py` - 备份

### 损失函数
- `losses/losses.py` - 当前版本（无需修改）
- `losses/losses_cpu.py` - 备份

### 测试脚本
- `test_gpu_training.py` - GPU + AMP 测试
- `test_gpu_fp32_training.py` - GPU FP32 测试

### 文档
- `README_GPU_SETUP.md` - GPU 设置指南
- `GPU_MIGRATION_SUMMARY.md` - 本文件

---

## 性能对比预期

| 配置 | 预期速度 | 显存占用 |
|------|----------|----------|
| CPU | 基准 | - |
| GPU FP32 | 5-10x | ~2-4 GB |
| GPU AMP | 8-15x | ~1.5-3 GB |

---

## 注意事项

1. **RTX 5060 兼容性**: 需要 PyTorch 2.7+ 或 nightly 版本
2. **显存管理**: 如 OOM，减小 `batch_size`
3. **cuDNN**: 自动启用 benchmark 模式
4. **回退机制**: CUDA 不可用时自动回退 CPU
