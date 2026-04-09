# fUS-VoxelMorph GPU/CPU 版本说明

## 文件说明

### 训练脚本

| 文件 | 说明 | 使用场景 |
|------|------|----------|
| `train_cpu.py` | CPU 版本（原始备份） | 无 GPU 时使用 |
| `train_gpu.py` | GPU 版本（推荐使用） | 有 NVIDIA GPU 时使用 |

### 模型文件

| 文件 | 说明 |
|------|------|
| `models/vxm2d.py` | GPU 优化版本（当前使用） |
| `models/vxm2d_cpu.py` | CPU 版本备份 |

### 损失函数

| 文件 | 说明 |
|------|------|
| `losses/losses.py` | GPU/CPU 通用版本（无需修改） |
| `losses/losses_cpu.py` | 备份文件 |

---

## GPU 版本特性

1. **自动混合精度 (AMP)**：使用 `torch.cuda.amp.autocast()` 和 `GradScaler` 加速训练
2. **非阻塞数据传输**：使用 `.to(device, non_blocking=True)` 加速数据加载
3. **cuDNN 优化**：自动启用 `torch.backends.cudnn.benchmark`
4. **持久化工作进程**：`persistent_workers=True` 加速数据加载
5. **显存监控**：实时打印和记录 GPU 显存占用

---

## 使用方法

### GPU 训练（推荐）

```bash
python train_gpu.py --config configs/default.yaml --epochs 200
```

### CPU 训练（备用）

```bash
python train_cpu.py --config configs/default.yaml --epochs 200
```

---

## 快速测试（5 epochs）

```bash
# GPU 测试
python test_gpu_training.py

# 或使用命令行
python train_gpu.py --synthetic --epochs 5 --batch_size 4
```

---

## 性能对比

在 RTX 5060 Laptop GPU 上测试：

- CPU 训练：~XX sec/epoch
- GPU 训练：~XX sec/epoch
- GPU + AMP：~XX sec/epoch（预计快 1.5-2x）

---

## 注意事项

1. GPU 版本会自动检测 CUDA 是否可用，如不可用会回退到 CPU
2. 混合精度训练在 2D 小模型上通常能加速 1.5-2 倍
3. 如出现显存不足，可减小 `batch_size`
4. 原 `train.py` 保持不变，可以使用但需要手动指定 `--device cuda`
