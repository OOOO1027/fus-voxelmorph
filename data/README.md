# fUS 数据加载管道

用于 fUS (functional ultrasound) Power Doppler 图像配准的完整数据加载和预处理管道。

## 功能特性

- **多格式支持**: `.npy`, `.npz`, `.mat` (MATLAB)
- **灵活预处理**: 归一化、log 变换、高斯平滑、resize/pad
- **数据增强**: 随机仿射变换、强度噪声、随机裁剪
- **合成数据**: 模拟血管树 + 已知变形场（用于训练和演示）

## 快速开始

### 1. 加载真实 fUS 数据

```python
from data import FUSDataset, FUSPairDataset, get_fus_transforms

# 基础数据集
dataset = FUSDataset(
    data_path='data/fus_frames/',      # 文件夹或 .npy/.mat 文件
    target_size=(128, 100),             # 标准 fUS 尺寸
    normalize='minmax',                 # 或 'percentile'
    log_transform=False,                # 对偏斜分布有用
    gaussian_sigma=None,                # 可选平滑
)

# 配准配对数据集（用于训练）
pair_dataset = FUSPairDataset(
    dataset,
    mode='consecutive',                 # 连续帧配对
    # mode='to_reference',              # 配准到参考帧
    # mode='sliding_window',            # 滑动窗口
)
```

### 2. 数据增强

```python
# 训练时启用增强
augmentation = get_fus_transforms(
    train=True,
    rotation=5,                         # ±5度旋转
    translation=5,                      # ±5像素平移
    noise_std=0.02                      # 强度噪声
)

dataset = FUSDataset(
    data_path='data/fus_frames/',
    target_size=(128, 100),
    normalize='minmax',
    augmentation=augmentation           # 应用增强
)
```

### 3. 使用合成数据

```python
from data import SyntheticFUSDataset

# 合成数据集（无限数据生成）
dataset = SyntheticFUSDataset(
    size=1000,                          # 样本数量
    image_size=(128, 100),
    motion_type='mixed',                # random | cardiac | breathing | mixed
    max_displacement=10,
    noise_level=0.05,
)

# DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 获取 batch
for source, target, flow_gt in dataloader:
    # source, target: (B, 1, H, W)
    # flow_gt: (B, 2, H, W) - ground truth 变形场
    pass
```

### 4. 生成合成数据集文件

```python
from data import create_synthetic_dataset

# 创建并保存合成数据集
create_synthetic_dataset(
    n_samples=100,
    save_dir='data/synthetic_train',
    image_size=(128, 100),
    motion_type='mixed',
)
```

## 配置文件

使用 `configs/data_pipeline.yaml` 配置数据管道：

```yaml
data:
  type: "real"                          # real | synthetic
  data_path: "data/fus_frames/"
  
  # 预处理
  target_size: [128, 100]
  normalize: "minmax"                   # minmax | percentile
  log_transform: false
  gaussian_sigma: null
  
  # 数据增强
  augmentation:
    enabled: true
    affine:
      rotation: 5
      translation: 5
    intensity:
      noise_std: 0.02
  
  # 配对策略
  pair_mode: "consecutive"
  ref_idx: 0
```

## 运行演示

```bash
# 运行所有数据管道演示
python data/demo_data_pipeline.py
```

这将生成多个可视化结果在 `demo_data/` 目录中：
- `demo1_real_data.png` - 真实数据加载
- `demo2_preprocessing.png` - 预处理对比
- `demo3_augmentation.png` - 数据增强
- `demo4_pairs.png` - 配对策略
- `demo5_synthetic.png` - 合成数据
- `demo6_dataloader.png` - DataLoader 使用

## 数据格式

### 输入格式

**文件夹模式:**
```
data/fus_frames/
  ├── frame_001.npy
  ├── frame_002.npy
  └── ...
```

**时间序列模式:**
```python
# 单个 .npy 文件，形状 (T, H, W) 或 (T, C, H, W)
data = np.load('fus_timeseries.npy')  # (100, 128, 100) - 100 帧
```

**MATLAB 格式:**
```matlab
% MATLAB
save('fus_data.mat', 'data');  % data 大小为 (T, H, W)
```

```python
# Python
dataset = FUSDataset('fus_data.mat', mat_key='data')
```

## 合成数据细节

### 血管树生成

使用分形分支算法生成模拟血管结构：
- 随机游走生成血管路径
- 随机分支模拟血管网络
- 高斯卷积模拟血管截面

### 变形场类型

1. **Random**: 基于控制点的随机平滑变形
2. **Cardiac**: 类似心跳的周期性径向变形
3. **Breathing**: 类似呼吸的上下平移+轻微缩放
4. **Mixed**: 随机混合以上类型

## 预处理选项

| 方法 | 适用场景 | 说明 |
|-----|---------|-----|
| `minmax` | 标准场景 | 简单归一化到 [0, 1] |
| `percentile` | 有异常值 | 基于 1-99 percentile 的鲁棒归一化 |
| `log_transform` | 动态范围大 | 压缩高信号值 |
| `gaussian_smooth` | 噪声大 | 空间平滑滤波 |

## 训练命令

```bash
# 使用真实数据训练
python train_v2.py --config configs/data_pipeline.yaml

# 使用合成数据训练
python train_v2.py --config configs/data_pipeline.yaml
# (修改 config 中 data.type 为 "synthetic")

# 命令行覆盖参数
python train_v2.py --config configs/data_pipeline.yaml \
    --data_path /path/to/data \
    --epochs 100 \
    --batch_size 8
```
