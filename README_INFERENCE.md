# fUS-VoxelMorph 推理与可视化

## 文件结构

```
.
├── register.py                  # 主推理脚本
├── visualize.py                 # 可视化工具
├── utils/
│   ├── visualization.py         # 可视化函数库
│   └── ...
├── scripts/
│   └── demo_visualization.py    # 可视化演示
├── INFERENCE.md                 # 详细使用文档
└── README_INFERENCE.md          # 本文件
```

## 快速开始

### 1. 单对图像配准

```bash
python register.py \
    --model checkpoints/best_model.pth \
    --source moving.npy \
    --target fixed.npy \
    --eval \
    --visualize \
    --output_dir results/
```

### 2. 时间序列配准

```bash
python register.py \
    --model checkpoints/best_model.pth \
    --data_path timeseries.npy \
    --ref_idx 0 \
    --eval \
    --visualize
```

### 3. 可视化已保存的结果

```bash
python visualize.py \
    --source source.npy \
    --target target.npy \
    --warped warped.npy \
    --flow flow.npy \
    --output_dir figures/
```

### 4. 演示所有可视化类型

```bash
python scripts/demo_visualization.py
```

## 可视化类型

| 可视化 | 说明 | 文件名 |
|-------|------|--------|
| Overview | 综合结果展示 | overview.png |
| Overlay | 绿/品红叠加对比 | overlay.png |
| Flow | 位移场（热力图+箭头） | flow.png |
| Jacobian | Jacobian 分析 | jacobian.png |
| Difference | 差异图 | difference.png |
| Grid | 变形网格 | grid.png |

## 评估指标

运行 `--eval` 时自动计算：
- **NCC/MSE/SSIM**: 图像相似度
- **Displacement**: 位移统计
- **Jacobian**: 变形折叠分析

完整文档请查看: [INFERENCE.md](INFERENCE.md)
