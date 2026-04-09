# fUS-VoxelMorph 训练指南

完整的训练脚本，支持真实数据和合成数据，包含 TensorBoard 日志和模型检查点。

## 快速开始

### 使用合成数据快速测试

```bash
# 运行快速演示（5 epochs，合成数据）
python scripts/quick_train_demo.py

# 或者手动运行
python train.py --synthetic --epochs 5 --batch_size 4
```

### 使用真实数据训练

```bash
# 准备数据
mkdir -p data/fus_frames/
# 将你的 .npy 文件放入该目录

# 开始训练
python train.py --config configs/default.yaml --data_path data/fus_frames/
```

## 命令行参数

### 数据选项
```bash
--data_path PATH           # 数据路径
--synthetic                # 使用合成数据
--synthetic_samples N      # 合成样本数量（默认 1000）
--pair_mode MODE           # 配对策略: consecutive | random | to_reference
--val_split RATIO          # 验证集比例（默认 0.1）
```

### 模型选项
```bash
--in_channels N            # 输入通道（默认 1）
--enc_channels N N N       # 编码器通道（默认 16 32 32 32）
--dec_channels N N N       # 解码器通道
--integration_steps N      # 积分步数（默认 7，0=无微分同胚约束）
```

### 损失选项
```bash
--similarity TYPE          # 相似度损失: ncc | mse（默认 ncc）
--ncc_win_size N           # NCC 窗口大小（默认 9）
--reg_type TYPE            # 正则化类型: grad | diffusion（默认 grad）
--reg_weight LAMBDA        # 正则化权重（默认 1.0）
--reg_penalty TYPE         # 正则化惩罚: l1 | l2（默认 l2）
```

### 训练选项
```bash
--epochs N                 # 训练轮数（默认 200）
--batch_size N             # 批次大小（默认 8）
--lr RATE                  # 学习率（默认 1e-4）
--weight_decay W           # 权重衰减（默认 0.0）
--lr_scheduler TYPE        # 学习率调度: cosine | step | plateau | exponential
--early_stopping N         # 早停耐心值（epochs）
```

### 数据增强
```bash
--augment                  # 启用数据增强
--rotation DEG             # 最大旋转角度（默认 5）
--translation PIX          # 最大平移像素（默认 5）
--noise_std STD            # 噪声标准差（默认 0.02）
```

### 日志和检查点
```bash
--save_dir DIR             # 检查点保存目录（默认 checkpoints）
--log_dir DIR              # TensorBoard 日志目录（默认 runs）
--log_interval N           # 每 N 个 batch 记录日志
--vis_interval N           # 每 N 个 batch 可视化
--save_interval N          # 每 N 个 epoch 保存检查点
--no_tensorboard           # 禁用 TensorBoard
--num_vis_samples N        # 可视化样本数量（默认 4）
```

### 系统选项
```bash
--device DEVICE            # 设备: cuda | cpu
--num_workers N            # 数据加载进程数（默认 4）
--seed N                   # 随机种子（默认 42）
```

### 恢复训练
```bash
--resume PATH              # 从检查点恢复
--start_epoch N            # 起始 epoch
```

## 配置文件的

创建 `configs/my_experiment.yaml`:

```yaml
# Data
data_path: "data/my_fus_data/"
synthetic: false
pair_mode: "consecutive"
val_split: 0.1

# Model
in_channels: 1
enc_channels: [16, 32, 32, 32]
dec_channels: [32, 32, 32, 32, 16, 16]
integration_steps: 7

# Loss
similarity: "ncc"
ncc_win_size: 9
reg_type: "grad"
reg_weight: 1.0

# Training
epochs: 300
batch_size: 16
lr: 1.0e-4
lr_scheduler: "cosine"

# Augmentation
augment: true
rotation: 5.0
translation: 5.0

# Logging
save_dir: "checkpoints/my_experiment"
log_dir: "runs/my_experiment"
use_tensorboard: true
```

然后运行：
```bash
python train.py --config configs/my_experiment.yaml
```

## 常用训练命令

### 1. 基础训练
```bash
python train.py \
    --data_path data/fus_frames/ \
    --epochs 200 \
    --batch_size 8 \
    --lr 1e-4 \
    --reg_weight 1.0
```

### 2. 使用数据增强
```bash
python train.py \
    --data_path data/fus_frames/ \
    --augment \
    --rotation 10 \
    --translation 10 \
    --noise_std 0.03
```

### 3. 调整正则化权重
```bash
# 更强的正则化（更平滑的变形场）
python train.py --reg_weight 2.0

# 更弱的正则化（更灵活的变形）
python train.py --reg_weight 0.5
```

### 4. 使用合成数据
```bash
python train.py \
    --synthetic \
    --synthetic_samples 5000 \
    --epochs 100 \
    --batch_size 16
```

### 5. 继续训练
```bash
python train.py \
    --config configs/default.yaml \
    --resume checkpoints/latest_checkpoint.pth \
    --epochs 300
```

### 6. 使用早停
```bash
python train.py \
    --early_stopping 20 \
    --epochs 500
```

## 监控训练

### TensorBoard

```bash
# 启动 TensorBoard
python scripts/launch_tensorboard.py

# 或手动启动
tensorboard --logdir=runs

# 访问 http://localhost:6006
```

TensorBoard 记录：
- 损失曲线（训练/验证）
- 学习率变化
- 配准结果可视化
- GPU 内存使用

### 控制台输出

训练时会显示：
```
[2026-04-05 20:00:00] [INFO] Epoch 10/200
  Batch [50/100] Loss: 0.1234 (avg: 0.1456) Sim: -0.2345 Reg: 0.0123
  Train - Loss: 0.1456 (Sim: -0.2345, Reg: 0.0123) Time: 25.3s
  Val   - Loss: 0.1567 (Sim: -0.2456, Reg: 0.0134) NCC: 0.8765 MSE: 0.0234 LR: 0.000095
```

## 输出文件

### 检查点目录 (`checkpoints/`)
```
checkpoints/
├── best_model.pth              # 验证集上表现最好的模型
├── latest_checkpoint.pth       # 最新检查点（用于恢复）
├── checkpoint_epoch20.pth      # 定期保存的检查点
├── checkpoint_epoch40.pth
└── training.log                # 训练日志
```

### TensorBoard 日志 (`runs/`)
```
runs/
└── 20260405_200000/            # 时间戳命名的实验
    ├── events.out.tfevents...  # 标量数据（损失、学习率）
    └── ...                     # 图像数据（配准可视化）
```

## 超参数调优建议

### 学习率
- **初始值**: 1e-4 (Adam)
- **太大 (>1e-3)**: 训练不稳定，可能发散
- **太小 (<1e-5)**: 收敛慢，可能陷入局部最优

### 正则化权重 (lambda)
- **从 1.0 开始**
- **太大 (>5.0)**: 变形场过于平滑，配准不准确
- **太小 (<0.1)**: 可能出现折叠（非微分同胚）
- **观察 Jacobian**: 如果负 Jacobian 比例 > 1%，增加 lambda

### Batch Size
- **2D fUS**: 8-16（内存允许的情况下）
- **更大 batch**: 更稳定的梯度，但需要更多内存
- **更小 batch**: 更频繁的更新，可能更好的泛化

### NCC 窗口大小
- **默认 9**: 适合标准 fUS (128x100)
- **更大 (11-15)**: 更平滑的相似性，适合大结构
- **更小 (5-7)**: 更精细的匹配，适合小血管

## 故障排除

### CUDA Out of Memory
```bash
# 减小 batch size
python train.py --batch_size 4

# 减小模型
python train.py --enc_channels 16 32 32 --dec_channels 32 32 32 32

# 使用 CPU
python train.py --device cpu
```

### 训练发散
```bash
# 减小学习率
python train.py --lr 5e-5

# 增加正则化
python train.py --reg_weight 2.0

# 检查数据归一化
```

### 验证损失不下降
```bash
# 启用数据增强
python train.py --augment

# 减小模型复杂度
python train.py --enc_channels 16 32 32

# 使用早停避免过拟合
python train.py --early_stopping 20
```

## 下一步

训练完成后，使用训练好的模型：

```bash
# 配准
python register.py --model checkpoints/best_model.pth --source src.npy --target tgt.npy

# 评估
python evaluate.py --model checkpoints/best_model.pth --data_path data/test/
```
