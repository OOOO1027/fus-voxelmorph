# fUS-VoxelMorph 推理与可视化指南

完整的推理脚本和可视化工具，支持多种输入格式和出版质量的可视化。

## 快速开始

### 单对图像配准

```bash
# 基本用法
python register.py \
    --model checkpoints/best_model.pth \
    --source moving.npy \
    --target fixed.npy \
    --output_dir results/

# 包含评估和可视化
python register.py \
    --model checkpoints/best_model.pth \
    --source moving.npy \
    --target fixed.npy \
    --eval \
    --visualize \
    --output_dir results/
```

### 时间序列配准

```bash
python register.py \
    --model checkpoints/best_model.pth \
    --data_path timeseries.npy \
    --ref_idx 0 \
    --batch_size 8 \
    --eval \
    --visualize \
    --output_dir results/
```

## register.py 命令行参数

### 模型参数
```bash
--model PATH              # 模型检查点路径（必需）
--integration_steps N     # 积分步数（默认 7）
--enc_channels N N N      # 编码器通道（默认 16 32 32 32）
--dec_channels N N N      # 解码器通道
```

### 输入参数
```bash
# 单对模式
--source PATH             # 移动图像路径
--target PATH             # 固定图像路径

# 时间序列模式
--data_path PATH          # 时间序列数据路径
--ref_idx N               # 参考帧索引（默认 0）
```

### 输出参数
```bash
--output_dir DIR          # 输出目录（默认 results/）
--prefix PREFIX           # 输出文件前缀
--save_npy                # 保存为 .npy 格式（默认）
--no_save_npy             # 不保存 .npy 文件
```

### 评估和可视化
```bash
--eval                    # 计算评估指标
--visualize, -v           # 生成可视化
--vis_format FORMAT       # 格式: png, pdf, svg（默认 png）
--vis_dpi N               # 图像 DPI（默认 150）
```

### 系统参数
```bash
--device DEVICE           # 设备: cuda 或 cpu（默认 cuda）
--batch_size N            # 时间序列批大小（默认 8）
```

## 支持的输入格式

| 格式 | 扩展名 | 说明 |
|-----|-------|-----|
| NumPy | .npy, .npz | 标准 NumPy 数组 |
| MATLAB | .mat | 需要 scipy |
| 图像 | .png, .jpg, .tiff | 需要 PIL |

## 输出文件

运行 `register.py` 后会生成：

```
results/
├── warped.npy              # 配准后的图像
├── flow.npy                # 位移场 (2, H, W)
├── source.npy              # 源图像（复制）
├── target.npy              # 目标图像（复制）
├── metrics.json            # 评估指标（--eval 时）
├── overview.png            # 综合结果图（--visualize 时）
├── overlay.png             # 叠加对比图
├── flow.png                # 位移场可视化
├── jacobian.png            # Jacobian 分析
├── difference.png          # 差异图
└── grid.png                # 变形网格图
```

## 评估指标

当使用 `--eval` 时，计算以下指标：

### 图像相似度
- **NCC (Normalized Cross-Correlation)**：-1 到 1，越高越好
- **MSE (Mean Squared Error)**：均方误差，越低越好
- **SSIM (Structural Similarity)**：结构相似度，0 到 1

### 变形场统计
- **Mean/Max Displacement**：平均/最大位移（像素）
- **Flow Std**：位移标准差

### Jacobian 分析
- **Mean det(J)**：平均 Jacobian 行列式（应接近 1）
- **% Folding**：负 Jacobian 比例（应 < 1%）

### 示例输出
```
============================================================
Registration Metrics
============================================================

Image Similarity:
  NCC  - Before: 0.6543  After: 0.9234  (Δ +0.2691)
  MSE  - Before: 0.2341  After: 0.0567  (Δ -0.1774)
  SSIM - After:  0.8912

Deformation Field:
  Mean displacement: 3.45 px
  Max displacement:  12.34 px
  Std displacement:  2.12 px

Jacobian Analysis:
  Mean det(J):  0.9876
  Min det(J):   0.2345
  |det|<=0:     0.12%
============================================================
```

## 可视化类型

### 1. 综合结果图 (overview.png)

包含：
- Source (Moving)
- Target (Fixed)
- Warped
- Overlay (Green/Magenta)
- Displacement magnitude

### 2. 叠加对比图 (overlay.png)

类似于 Zhong et al. Fig.3 的风格：
- **Before**: Green/Magenta 显示配准前的差异
- **After**: Green/Magenta 显示配准后的对齐
- **Improvement**: 颜色变化表示配准效果

### 3. 位移场可视化 (flow.png)

包含：
- **Magnitude heatmap**: 位移大小的热力图
- **Direction map**: 位移方向的色相图
- **Quiver plot**: 位移向量的箭头图

### 4. Jacobian 分析 (jacobian.png)

包含：
- **det(J) heatmap**: Jacobian 行列式空间分布
- **Folding regions**: 变形折叠区域（红色标记）
- **Histogram**: Jacobian 值分布直方图

### 5. 差异图 (difference.png)

包含：
- **|Moving - Fixed|**: 配准前差异
- **|Warped - Fixed|**: 配准后差异
- **Improvement map**: 改进区域（绿：变好，红：变差）
- **Histogram**: 差异分布

### 6. 变形网格 (grid.png)

显示：
- **Regular grid**: 原始规则网格
- **Deformed grid**: 变形后的网格

## visualize.py 工具

用于重新可视化已保存的结果：

```bash
# 基本用法
python visualize.py \
    --source source.npy \
    --target target.npy \
    --warped warped.npy \
    --flow flow.npy \
    --output_dir figures/

# 指定特定可视化类型
python visualize.py \
    --source src.npy --target tgt.npy --warped warped.npy --flow flow.npy \
    --only overview overlay flow \
    --output_dir figures/

# 高分辨率输出
python visualize.py \
    --source src.npy --target tgt.npy --warped warped.npy --flow flow.npy \
    --format pdf --dpi 300 \
    --output_dir figures/

# 可视化时间序列
python visualize.py \
    --series warped_series.npy \
    --flows flows.npy \
    --ref_idx 0 \
    --frame_idx 10 \
    --output_dir figures/
```

### visualize.py 参数

```bash
# 输入文件
--source PATH             # 源图像
--target PATH             # 目标图像
--warped PATH             # 配准结果
--flow PATH               # 位移场
--metrics PATH            # 指标 JSON 文件

# 时间序列
--series PATH             # 配准后的时间序列
--flows PATH              # 位移场时间序列
--ref_idx N               # 参考帧索引
--frame_idx N             # 可视化的帧索引

# 输出选项
-o, --output_dir DIR      # 输出目录
--prefix PREFIX           # 文件前缀
--format FORMAT           # 格式: png, pdf, svg, jpg
--dpi N                   # DPI（默认 300）

# 可视化选择
--only TYPE [TYPE ...]    # 选择特定类型:
                          # overview, overlay, flow,
                          # jacobian, difference, grid, all
--no_metrics              # 不显示指标
--title TITLE             # 自定义标题
```

## 演示脚本

```bash
# 生成合成数据并展示所有可视化类型
python scripts/demo_visualization.py

# 输出保存在 demo_figures/ 目录
```

## 常见用法示例

### 1. 完整推理流程

```bash
# 训练模型
python train.py --config configs/default.yaml

# 配准单对图像
python register.py \
    --model checkpoints/best_model.pth \
    --source data/test/moving_001.npy \
    --target data/test/fixed.npy \
    --eval \
    --visualize \
    --output_dir results/pair_001/

# 查看结果
ls results/pair_001/
# metrics.json  overview.png  warped.npy  ...
```

### 2. 批量处理时间序列

```bash
# 配准整个时间序列
python register.py \
    --model checkpoints/best_model.pth \
    --data_path data/experiment/timeseries.npy \
    --ref_idx 10 \
    --batch_size 16 \
    --eval \
    --visualize \
    --output_dir results/experiment/

# 查看特定帧
python visualize.py \
    --series results/experiment/warped_series.npy \
    --flows results/experiment/flows.npy \
    --ref_idx 10 \
    --frame_idx 25 \
    --dpi 300 \
    --output_dir figures/frame_25/
```

### 3. 对比实验

```bash
# 不同正则化权重的对比
for lambda in 0.5 1.0 2.0; do
    python register.py \
        --model checkpoints/lambda_${lambda}/best_model.pth \
        --source test/source.npy \
        --target test/target.npy \
        --eval \
        --prefix lambda_${lambda} \
        --output_dir comparison/
done

# 对比可视化
python visualize.py \
    --source test/source.npy --target test/target.npy \
    --warped comparison/lambda_0.5_warped.npy \
    --flow comparison/lambda_0.5_flow.npy \
    --prefix lambda_0.5 \
    --output_dir comparison/figures/
```

### 4. 生成论文图表

```bash
# 高分辨率 PDF 输出
python register.py \
    --model checkpoints/best_model.pth \
    --source fig2a.npy --target fig2b.npy \
    --visualize \
    --vis_format pdf \
    --vis_dpi 600 \
    --output_dir paper_figures/figure2/

# 或使用 visualize.py 重新生成
python visualize.py \
    --source fig2a.npy --target fig2b.npy \
    --warped warped.npy --flow flow.npy \
    --only overview overlay \
    --format pdf --dpi 600 \
    --title "Figure 2: Registration Result" \
    --output_dir paper_figures/figure2/
```

## 故障排除

### CUDA Out of Memory

```bash
# 减小批大小
python register.py --model model.pth --data_path data.npy --batch_size 4

# 或使用 CPU
python register.py --model model.pth --data_path data.npy --device cpu
```

### 图像尺寸不匹配

```bash
# register.py 会自动 resize 输入图像
# 确保训练时和推理时的尺寸一致
```

### 可视化质量问题

```bash
# 增加 DPI
python register.py ... --vis_dpi 300

# 或使用矢量格式
python register.py ... --vis_format pdf
```

## 编程接口

Python 代码中使用可视化函数：

```python
from utils.visualization import (
    create_registration_figure,
    create_overlay_comparison,
    save_all_visualizations
)

# 创建特定可视化
fig = create_registration_figure(source, target, warped, flow)
fig.savefig('result.png', dpi=150)

# 生成所有可视化
save_all_visualizations(
    source, target, warped, flow,
    output_dir='figures/',
    prefix='exp1_',
    format='png',
    dpi=150
)
```
