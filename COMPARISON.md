# fUS-VoxelMorph 与传统方法对比

对比实验框架，评估 VoxelMorph 与经典配准方法的性能差异。

## 传统 Baseline 方法

### 1. 刚体配准 (Rigid Registration)
- **变换类型**: 旋转 + 平移 (3 DOF)
- **实现**: SimpleITK Euler2DTransform
- **适用场景**: 只有刚性运动，无变形
- **优点**: 计算快，结果稳定
- **缺点**: 无法处理形变

### 2. 仿射配准 (Affine Registration)
- **变换类型**: 旋转 + 平移 + 缩放 + 剪切 (6 DOF)
- **实现**: SimpleITK AffineTransform
- **适用场景**: 全局线性变形
- **优点**: 比刚体更灵活
- **缺点**: 无法处理局部非线性变形

### 3. Demons 算法
- **类型**: 光学流驱动的可变形配准
- **实现**: SimpleITK DemonsRegistrationFilter
- **适用场景**: 平滑的可变形配准
- **优点**: 计算较快，适合小变形
- **缺点**: 对噪声敏感，可能产生不规则变形

### 4. B-spline 配准
- **类型**: 基于自由形变 (FFD) 的可变形配准
- **实现**: SimpleITK BSplineTransform
- **适用场景**: 复杂的局部变形
- **优点**: 产生平滑的变形场
- **缺点**: 计算慢，需要调参

## 对比指标 (参考 Zhong et al.)

### 图像相似度
| 指标 | 范围 | 说明 |
|-----|------|-----|
| **NCC** | [-1, 1] | 归一化互相关，越高越好 |
| **MS-SSIM** | [0, 1] | 多尺度结构相似度，越高越好 |
| **DSC** | [0, 1] | Dice 相似系数（二值掩模），越高越好 |

### 变形场分析
| 指标 | 说明 |
|-----|------|
| **Jac Mean** | Jacobian 行列式均值（应接近 1） |
| **Jac %Neg** | 负 Jacobian 比例（折叠率，应 < 1%） |
| **Flow Mean** | 平均位移（像素） |
| **Flow Max** | 最大位移（像素） |

### 计算效率
| 指标 | 说明 |
|-----|------|
| **Time (s)** | 单对图像配准时间 |

## 使用方法

### 安装依赖

```bash
# SimpleITK 用于传统方法
pip install SimpleITK

# pandas 用于结果表格
pip install pandas openpyxl
```

### 快速演示

```bash
# 运行简化版对比（使用合成数据）
python scripts/demo_comparison.py
```

### 完整对比实验

```bash
# 使用合成数据
python compare_baseline.py \
    --synthetic \
    --n_samples 50 \
    --model checkpoints/best_model.pth \
    --output_dir comparison/

# 使用真实数据
python compare_baseline.py \
    --data_path data/test/ \
    --model checkpoints/best_model.pth \
    --n_samples 30 \
    --output_dir comparison/
```

### 选择对比的方法

```bash
# 只对比特定方法
python compare_baseline.py \
    --synthetic \
    --methods rigid affine voxelmorph \
    --model checkpoints/best_model.pth \
    --output_dir comparison/
```

### 调整传统方法参数

```bash
# 调整 Demons 和 B-spline 参数
python compare_baseline.py \
    --synthetic \
    --demons_iterations 200 \
    --bspline_grid 15 15 \
    --bspline_iterations 200 \
    --model checkpoints/best_model.pth \
    --output_dir comparison/
```

## 命令行参数

```bash
# 数据选项
--data_path PATH          # 真实数据路径
--synthetic               # 使用合成数据
--n_samples N             # 测试样本数（默认 20）
--image_size H W          # 图像尺寸（默认 128 100）

# 模型选项
--model PATH              # VoxelMorph 模型路径
--integration_steps N     # 积分步数（默认 7）

# 方法选择
--methods METHOD [METHOD ...]  # 对比的方法列表
                               # 可选: rigid, affine, demons, bspline, voxelmorph

# 传统方法参数
--demons_iterations N     # Demons 迭代次数（默认 100）
--bspline_grid X Y        # B-spline 控制点网格（默认 10 10）
--bspline_iterations N    # B-spline 优化迭代（默认 100）

# 输出选项
--output_dir DIR          # 输出目录（默认 comparison_results）
--save_visualizations     # 保存每个样本的可视化

# 系统选项
--device DEVICE           # 设备: cuda 或 cpu
--seed N                  # 随机种子
```

## 输出文件

```
comparison_results/
├── comparison_table.csv     # 对比表格 (CSV)
├── comparison_table.xlsx    # 对比表格 (Excel)
├── metrics_comparison.png   # 指标对比图
├── raw_results.json         # 原始结果数据
└── summary.txt              # 文字总结
```

## 结果解读

### 示例输出

```
======================================================================
                        Registration Methods Comparison
======================================================================
Test Case  Method      NCC (after)  NCC Improvement  MS-SSIM    DSC    Time (s)
======================================================================
pair_000   Rigid          0.7234         +0.0892      0.8234   0.7123   0.234
pair_000   Affine         0.8123         +0.1781      0.8912   0.8234   0.456
pair_000   Demons         0.8912         +0.2570      0.9234   0.8654   1.234
pair_000   B-spline       0.9012         +0.2670      0.9345   0.8789   2.567
pair_000   VoxelMorph     0.9234         +0.2892      0.9567   0.9012   0.089
...
Average    Rigid          0.7345         +0.0912      0.8345   0.7234   0.245
Average    Affine         0.8234         +0.1801      0.9012   0.8345   0.478
Average    Demons         0.8891         +0.2458      0.9301   0.8712   1.312
Average    B-spline       0.8989         +0.2556      0.9389   0.8801   2.678
Average    VoxelMorph     0.9201         +0.2768      0.9523   0.8956   0.092
======================================================================

Key Findings:
======================================================================

Best NCC (after):  VoxelMorph = 0.9201
Best MS-SSIM:      VoxelMorph = 0.9523
Best DSC:          VoxelMorph = 0.8956
Fastest:           VoxelMorph = 0.092s
Lowest folding:    VoxelMorph = 0.15%
======================================================================
```

### 典型结论

1. **准确性**: VoxelMorph 通常优于传统方法（更高的 NCC/SSIM/DSC）
2. **速度**: VoxelMorph 推理最快（GPU 加速）
3. **平滑性**: B-spline 和 VoxelMorph 产生最平滑的变形场
4. **适用性**:
   - 刚体/仿射: 适合小变形，计算快
   - Demons: 适合中等变形，参数敏感
   - B-spline: 适合大变形，计算慢
   - VoxelMorph: 适合各种变形，需要训练

## 可视化

生成的 `metrics_comparison.png` 包含：
- NCC (after)
- NCC Improvement
- MS-SSIM
- DSC
- Jacobian Folding %
- Inference Time

每个指标一个子图，不同方法用不同颜色显示。

## 编程接口

Python 代码中使用：

```python
from baselines import (
    RigidRegistration,
    AffineRegistration,
    compare_methods
)

# 创建传统方法
rigid = RigidRegistration(metric='mean_squares', max_iterations=200)
affine = AffineRegistration(metric='mean_squares', max_iterations=200)

# 配准
warped, params = rigid.register(source, target)
flow = rigid.get_displacement_field(source.shape)

# 对比多种方法
methods = {
    'Rigid': rigid,
    'Affine': affine,
    'VoxelMorph': model,
}

results = compare_methods(source, target, methods, device='cuda')
```

## 注意事项

1. **SimpleITK 安装**:
   - Windows: `pip install SimpleITK`
   - Linux/Mac: `pip install SimpleITK`
   - 可能需要 `pip install --upgrade pip` 先

2. **计算时间**:
   - 传统方法（特别是 B-spline）可能需要几分钟
   - VoxelMorph GPU 推理通常 < 0.1s

3. **内存使用**:
   - B-spline 高分辨率网格需要较多内存
   - 如 OOM，减小 `--bspline_grid`

4. **参数调优**:
   - 不同数据集可能需要调整传统方法参数
   - VoxelMorph 参数固定（训练后）

## 引用

对比指标参考：
- Zhong et al. "Real-time Whole-brain Functional Ultrasound Imaging..."
- Balakrishnan et al. "VoxelMorph: A Learning Framework for Deformable Medical Image Registration"
