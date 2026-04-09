"""
合成 fUS 数据生成器。

用于在没有真实 fUS 数据时进行训练和演示。

功能：
1. 生成模拟血管树图案（分形结构）
2. 生成已知的变形场作为 ground truth
3. 应用变形场生成配准对 (source, target, flow_gt)

血管树生成方法：
- 基于随机游走和分支的生长模型
- 模拟真实血管的管状结构
- 添加噪声模拟血流信号
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


# -----------------------------------------------------------------------------
#  血管树生成
# -----------------------------------------------------------------------------

def generate_vessel_tree(size=(128, 100), n_vessels=8, branch_prob=0.3,
                         thickness_range=(1, 4), intensity_range=(0.5, 1.0),
                         seed=None):
    """
    生成模拟血管树图案。

    使用随机游走+分支的算法生成类似血管的结构。

    Parameters
    ----------
    size : tuple
        输出尺寸 (H, W)，默认 (128, 100)
    n_vessels : int
        初始血管数量，默认 8
    branch_prob : float
        分支概率，默认 0.3
    thickness_range : tuple
        血管粗细范围，默认 (1, 4)
    intensity_range : tuple
        信号强度范围，默认 (0.5, 1.0)
    seed : int or None
        随机种子

    Returns
    -------
    np.ndarray : 血管图像，形状 (H, W)，值域 [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)

    H, W = size
    image = np.zeros((H, W), dtype=np.float32)

    def draw_line(img, p1, p2, thickness, intensity):
        """在图像上绘制线段（血管段）。"""
        x0, y0 = int(p1[0]), int(p1[1])
        x1, y1 = int(p2[0]), int(p2[1])

        # Bresenham 线段算法
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            # 绘制圆形区域模拟血管粗细
            for dx_t in range(-thickness, thickness + 1):
                for dy_t in range(-thickness, thickness + 1):
                    if dx_t * dx_t + dy_t * dy_t <= thickness * thickness:
                        x, y = x0 + dx_t, y0 + dy_t
                        if 0 <= x < W and 0 <= y < H:
                            # 高斯衰减模拟血管截面
                            dist = np.sqrt(dx_t * dx_t + dy_t * dy_t)
                            val = intensity * np.exp(-dist * dist / (2 * thickness * thickness / 4))
                            img[y, x] = max(img[y, x], val)

            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def grow_vessel(start_pos, start_angle, length, thickness, intensity):
        """生长一条血管（带分支）。"""
        pos = np.array(start_pos, dtype=np.float32)
        angle = start_angle
        segment_length = 5
        positions = [pos.copy()]

        for _ in range(int(length / segment_length)):
            # 随机改变方向（模拟血管弯曲）
            angle += np.random.normal(0, 0.15)

            # 计算新位置
            new_pos = pos + segment_length * np.array([np.cos(angle), np.sin(angle)])

            # 边界检查
            if not (0 <= new_pos[0] < W and 0 <= new_pos[1] < H):
                break

            # 绘制血管段
            draw_line(image, pos, new_pos, thickness, intensity)

            positions.append(new_pos.copy())
            pos = new_pos

            # 分支
            if np.random.random() < branch_prob and thickness > 1:
                branch_angle = angle + np.random.choice([-1, 1]) * np.random.uniform(0.3, 0.8)
                branch_length = length * np.random.uniform(0.4, 0.7)
                branch_thickness = max(1, thickness - 1)
                grow_vessel(pos.copy(), branch_angle, branch_length, branch_thickness,
                           intensity * np.random.uniform(0.7, 0.95))

    # 生成多个初始血管
    for i in range(n_vessels):
        # 随机起始位置（从边缘开始）
        side = np.random.randint(4)  # 0: 上, 1: 右, 2: 下, 3: 左
        if side == 0:  # 上
            start = (np.random.uniform(0, W), 0)
            angle = np.random.uniform(0.3, 2.8)  # 向下
        elif side == 1:  # 右
            start = (W - 1, np.random.uniform(0, H))
            angle = np.random.uniform(1.8, 4.5)  # 向左
        elif side == 2:  # 下
            start = (np.random.uniform(0, W), H - 1)
            angle = np.random.uniform(3.5, 5.8)  # 向上
        else:  # 左
            start = (0, np.random.uniform(0, H))
            angle = np.random.uniform(-1.2, 1.2)  # 向右

        length = np.random.uniform(50, 100)
        thickness = np.random.randint(*thickness_range)
        intensity = np.random.uniform(*intensity_range)

        grow_vessel(start, angle, length, thickness, intensity)

    # 后处理：添加背景噪声和轻微模糊
    background = np.random.normal(0.05, 0.02, (H, W))
    image = np.clip(image + background, 0, 1)
    image = gaussian_filter(image, sigma=0.5)

    return image


def generate_vessel_network(size=(128, 100), density=0.02, n_branches=50, seed=None):
    """
    生成更复杂的血管网络（中心放射状+分支）。

    Parameters
    ----------
    size : tuple
        输出尺寸 (H, W)
    density : float
        血管密度参数，默认 0.02
    n_branches : int
        分支数量，默认 50
    seed : int or None
        随机种子

    Returns
    -------
    np.ndarray : 血管网络图像 (H, W)
    """
    if seed is not None:
        np.random.seed(seed)

    H, W = size
    image = np.zeros((H, W), dtype=np.float32)

    # 生成几个主要血管中心
    n_centers = np.random.randint(3, 6)
    centers = [(np.random.uniform(W * 0.2, W * 0.8),
                np.random.uniform(H * 0.2, H * 0.8)) for _ in range(n_centers)]

    for cx, cy in centers:
        # 从中心向外放射状生成血管
        n_rays = np.random.randint(5, 10)
        for r in range(n_rays):
            angle = 2 * np.pi * r / n_rays + np.random.normal(0, 0.2)
            length = np.random.uniform(30, 80)
            thickness = np.random.randint(2, 5)
            intensity = np.random.uniform(0.6, 1.0)

            # 血管路径
            n_points = int(length / 2)
            points = []
            for i in range(n_points):
                t = i / n_points
                # 添加弯曲
                curve_angle = angle + np.sin(t * np.pi * 2) * 0.3
                x = cx + t * length * np.cos(curve_angle)
                y = cy + t * length * np.sin(curve_angle)
                points.append((x, y))

            # 绘制血管
            for i in range(len(points) - 1):
                p1, p2 = points[i], points[i + 1]
                # 绘制线段（简化版）
                x_coords = np.linspace(p1[0], p2[0], 20)
                y_coords = np.linspace(p1[1], p2[1], 20)
                for x, y in zip(x_coords, y_coords):
                    x, y = int(x), int(y)
                    if 0 <= x < W and 0 <= y < H:
                        y_grid, x_grid = np.mgrid[max(0, y - thickness):min(H, y + thickness + 1),
                                                   max(0, x - thickness):min(W, x + thickness + 1)]
                        dist = np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)
                        mask = dist <= thickness
                        values = intensity * np.exp(-dist[mask] ** 2 / (2 * (thickness / 2) ** 2))
                        image[y_grid[mask], x_grid[mask]] = np.maximum(
                            image[y_grid[mask], x_grid[mask]], values)

    # 添加微小血管（背景）
    n_tiny = 100
    for _ in range(n_tiny):
        x = np.random.randint(0, W)
        y = np.random.randint(0, H)
        intensity = np.random.uniform(0.1, 0.3)
        image[y, x] = max(image[y, x], intensity)

    # 平滑
    image = gaussian_filter(image, sigma=0.8)

    # 添加噪声
    noise = np.random.normal(0, 0.03, (H, W))
    image = np.clip(image + noise, 0, 1)

    return image


# -----------------------------------------------------------------------------
#  变形场生成
# -----------------------------------------------------------------------------

def generate_deformation_field(size=(128, 100), n_control_points=5,
                               max_displacement=10, smooth_sigma=None,
                               seed=None):
    """
    生成平滑的变形场（位移场）。

    使用基于控制点的薄板样条/网格插值方法生成平滑变形。

    Parameters
    ----------
    size : tuple
        图像尺寸 (H, W)
    n_control_points : int
        控制点网格分辨率，默认 5
    max_displacement : float
        最大位移像素数，默认 10
    smooth_sigma : float or None
        高斯平滑 sigma，默认 None（自动计算）
    seed : int or None
        随机种子

    Returns
    -------
    flow : np.ndarray
        位移场，形状 (2, H, W)，[dx, dy]
    """
    if seed is not None:
        np.random.seed(seed)

    H, W = size

    if smooth_sigma is None:
        smooth_sigma = max(H, W) / n_control_points

    # 生成稀疏控制点位移
    grid_h = np.linspace(0, H - 1, n_control_points)
    grid_w = np.linspace(0, W - 1, n_control_points)

    # 随机位移
    disp_h = np.random.uniform(-max_displacement, max_displacement,
                               (n_control_points, n_control_points))
    disp_w = np.random.uniform(-max_displacement, max_displacement,
                               (n_control_points, n_control_points))

    # 插值到完整尺寸
    from scipy.interpolate import RegularGridInterpolator

    interp_h = RegularGridInterpolator((grid_h, grid_w), disp_h,
                                       bounds_error=False, fill_value=0)
    interp_w = RegularGridInterpolator((grid_h, grid_w), disp_w,
                                       bounds_error=False, fill_value=0)

    y_coords, x_coords = np.mgrid[0:H, 0:W]
    points = np.stack([y_coords.ravel(), x_coords.ravel()], axis=-1)

    flow_h = interp_h(points).reshape(H, W)
    flow_w = interp_w(points).reshape(H, W)

    # 平滑变形场
    flow_h = gaussian_filter(flow_h, sigma=smooth_sigma)
    flow_w = gaussian_filter(flow_w, sigma=smooth_sigma)

    # 组合成 (2, H, W) 格式 [dx, dy]
    flow = np.stack([flow_w, flow_h], axis=0).astype(np.float32)

    return flow


def generate_cardiac_like_motion(size=(128, 100), n_cycles=2, phase=0, seed=None):
    """
    生成类似心跳的周期性变形。

    模拟心脏收缩-舒张的周期性运动模式。

    Parameters
    ----------
    size : tuple
        图像尺寸 (H, W)
    n_cycles : float
        周期数，默认 2
    phase : float
        相位 [0, 2π]，默认 0
    seed : int or None
        随机种子

    Returns
    -------
    flow : np.ndarray
        位移场 (2, H, W)
    """
    if seed is not None:
        np.random.seed(seed)

    H, W = size
    y, x = np.mgrid[0:H, 0:W]

    # 中心点
    cy, cx = H / 2, W / 2

    # 径向距离和角度
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    theta = np.arctan2(y - cy, x - cx)

    # 周期性径向变形（心跳样）
    max_r = np.sqrt((H / 2) ** 2 + (W / 2) ** 2)
    normalized_r = r / max_r

    # 心跳波形：收缩期快，舒张期慢
    cardiac_phase = np.sin(phase + normalized_r * n_cycles * 2 * np.pi)
    displacement_magnitude = 5 * cardiac_phase * (1 - normalized_r)  # 中心大，边缘小

    # 添加旋转分量
    rotation = 2 * np.sin(phase / 2) * (1 - normalized_r)

    # 计算位移
    dr = displacement_magnitude
    dtheta = rotation * np.pi / 180

    flow_x = dr * np.cos(theta + dtheta) - (x - cx)
    flow_y = dr * np.sin(theta + dtheta) - (y - cy)

    # 平滑
    flow_x = gaussian_filter(flow_x, sigma=3)
    flow_y = gaussian_filter(flow_y, sigma=3)

    flow = np.stack([flow_x, flow_y], axis=0).astype(np.float32)

    return flow


def generate_breathing_motion(size=(128, 100), amplitude=5, frequency=1, phase=0):
    """
    生成类似呼吸的变形（主要是上下方向的平移+轻微缩放）。

    Parameters
    ----------
    size : tuple
        图像尺寸 (H, W)
    amplitude : float
        最大位移幅度，默认 5 像素
    frequency : float
        频率参数，默认 1
    phase : float
        相位 [0, 2π]

    Returns
    -------
    flow : np.ndarray
        位移场 (2, H, W)
    """
    H, W = size
    y, x = np.mgrid[0:H, 0:W]

    # 上下平移为主
    flow_y = amplitude * np.sin(phase) * np.ones_like(y)

    # 轻微的水平缩放（模拟呼吸时胸廓扩张）
    cx = W / 2
    flow_x = 0.02 * amplitude * np.sin(phase) * (x - cx)

    # 平滑
    flow_x = gaussian_filter(flow_x, sigma=2)
    flow_y = gaussian_filter(flow_y, sigma=2)

    flow = np.stack([flow_x, flow_y], axis=0).astype(np.float32)

    return flow


# -----------------------------------------------------------------------------
#  图像变形
# -----------------------------------------------------------------------------

def apply_deformation(image, flow, mode='bilinear'):
    """
    使用变形场对图像进行变形。

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        输入图像 (H, W) 或 (C, H, W) 或 (B, C, H, W)
    flow : np.ndarray or torch.Tensor
        位移场 (2, H, W) [dx, dy]
    mode : str
        插值模式

    Returns
    -------
    warped : np.ndarray or torch.Tensor
        变形后的图像
    """
    is_numpy = isinstance(image, np.ndarray)

    if is_numpy:
        image = torch.from_numpy(image).float()
        flow = torch.from_numpy(flow).float()

    # 确保是 4D (B, C, H, W)
    orig_ndim = image.ndim
    if orig_ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif orig_ndim == 3:
        image = image.unsqueeze(0)  # (1, C, H, W)

    B, C, H, W = image.shape

    # 归一化位移场到 [-1, 1] 用于 grid_sample
    flow_norm = flow.clone()
    flow_norm[0] = flow_norm[0] * 2.0 / (W - 1)  # dx
    flow_norm[1] = flow_norm[1] * 2.0 / (H - 1)  # dy

    # 创建网格
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1).to(flow.device)  # (H, W, 2)

    # 添加位移
    flow_permuted = flow_norm.permute(1, 2, 0)  # (H, W, 2)
    sample_grid = base_grid + flow_permuted
    sample_grid = sample_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

    # 采样
    warped = F.grid_sample(image, sample_grid, mode=mode,
                           padding_mode='border', align_corners=True)

    # 恢复维度
    if orig_ndim == 2:
        warped = warped[0, 0]
    elif orig_ndim == 3:
        warped = warped[0]

    if is_numpy:
        warped = warped.numpy()

    return warped


# -----------------------------------------------------------------------------
#  合成数据集
# -----------------------------------------------------------------------------

class SyntheticFUSGenerator:
    """
    合成 fUS 数据生成器。

    生成配准训练数据：(source, target, flow_gt)

    Parameters
    ----------
    size : tuple
        图像尺寸 (H, W)，默认 (128, 100)
    motion_type : str
        运动类型: 'random', 'cardiac', 'breathing', 'mixed'
    max_displacement : float
        最大位移，默认 10
    noise_level : float
        噪声水平，默认 0.05
    """

    def __init__(self, size=(128, 100), motion_type='mixed',
                 max_displacement=10, noise_level=0.05):
        self.size = size
        self.motion_type = motion_type
        self.max_displacement = max_displacement
        self.noise_level = noise_level

    def generate_pair(self, seed=None):
        """
        生成一对配准数据。

        Returns
        -------
        source : np.ndarray (H, W)
        target : np.ndarray (H, W)
        flow_gt : np.ndarray (2, H, W) - ground truth 位移场
        """
        if seed is not None:
            np.random.seed(seed)

        # 生成源图像（血管树）
        source = generate_vessel_tree(
            size=self.size,
            n_vessels=np.random.randint(5, 12),
            seed=seed
        )

        # 添加噪声
        source = source + np.random.normal(0, self.noise_level, self.size)
        source = np.clip(source, 0, 1)

        # 选择运动类型
        motion = self.motion_type
        if motion == 'mixed':
            motion = np.random.choice(['random', 'cardiac', 'breathing'])

        # 生成变形场
        if motion == 'random':
            flow_gt = generate_deformation_field(
                size=self.size,
                n_control_points=np.random.randint(3, 7),
                max_displacement=self.max_displacement,
                seed=seed
            )
        elif motion == 'cardiac':
            phase = np.random.uniform(0, 2 * np.pi)
            flow_gt = generate_cardiac_like_motion(
                size=self.size,
                n_cycles=np.random.uniform(1, 3),
                phase=phase,
                seed=seed
            )
        elif motion == 'breathing':
            phase = np.random.uniform(0, 2 * np.pi)
            flow_gt = generate_breathing_motion(
                size=self.size,
                amplitude=self.max_displacement,
                phase=phase
            )
        else:
            raise ValueError(f"Unknown motion type: {motion}")

        # 应用变形生成 target
        target = apply_deformation(source, flow_gt)

        # 为目标添加独立噪声
        target = target + np.random.normal(0, self.noise_level, self.size)
        target = np.clip(target, 0, 1)

        return source, target, flow_gt

    def generate_batch(self, batch_size, seed=None):
        """
        生成一批数据。

        Returns
        -------
        sources : np.ndarray (B, 1, H, W)
        targets : np.ndarray (B, 1, H, W)
        flows : np.ndarray (B, 2, H, W)
        """
        sources = []
        targets = []
        flows = []

        for i in range(batch_size):
            s, t, f = self.generate_pair(seed=None if seed is None else seed + i)
            sources.append(s[None])  # (1, H, W)
            targets.append(t[None])
            flows.append(f)

        sources = np.stack(sources, axis=0)  # (B, 1, H, W)
        targets = np.stack(targets, axis=0)
        flows = np.stack(flows, axis=0)

        return sources, targets, flows


class SyntheticFUSDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset 包装器，用于合成 fUS 数据。

    可用于无限数据生成（适合训练）。

    Parameters
    ----------
    size : int
        数据集大小（迭代次数）
    image_size : tuple
        图像尺寸 (H, W)
    motion_type : str
        运动类型
    max_displacement : float
        最大位移
    """

    def __init__(self, size=1000, image_size=(128, 100), motion_type='mixed',
                 max_displacement=10, noise_level=0.05, pregenerate=True):
        self.size = size
        self.generator = SyntheticFUSGenerator(
            size=image_size,
            motion_type=motion_type,
            max_displacement=max_displacement,
            noise_level=noise_level
        )

        # 预生成所有数据到内存（避免每次 __getitem__ 重新生成）
        self._cache = None
        if pregenerate:
            print(f"  Pre-generating {size} synthetic pairs...")
            sources, targets, flows = [], [], []
            for i in range(size):
                s, t, f = self.generator.generate_pair(seed=i)
                sources.append(s)
                targets.append(t)
                flows.append(f)
                if (i + 1) % 100 == 0:
                    print(f"    {i+1}/{size} generated")
            self._cache = (
                np.stack(sources),  # (N, H, W)
                np.stack(targets),
                np.stack(flows),    # (N, 2, H, W)
            )
            print(f"  Synthetic data cached in memory.")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self._cache is not None:
            source = self._cache[0][idx]
            target = self._cache[1][idx]
            flow_gt = self._cache[2][idx]
        else:
            source, target, flow_gt = self.generator.generate_pair(seed=idx)

        # 转换为 tensor 并添加通道维度
        source = torch.from_numpy(source[None].copy()).float()  # (1, H, W)
        target = torch.from_numpy(target[None].copy()).float()
        flow_gt = torch.from_numpy(flow_gt.copy()).float()  # (2, H, W)

        return source, target, flow_gt


# -----------------------------------------------------------------------------
#  工具函数
# -----------------------------------------------------------------------------

def save_synthetic_sample(source, target, flow_gt, save_dir, idx=0):
    """
    保存合成样本用于可视化。

    Parameters
    ----------
    source, target, flow_gt : np.ndarray
        图像和变形场
    save_dir : str
        保存目录
    idx : int
        样本索引
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # 保存为 npy
    np.save(os.path.join(save_dir, f'source_{idx}.npy'), source)
    np.save(os.path.join(save_dir, f'target_{idx}.npy'), target)
    np.save(os.path.join(save_dir, f'flow_gt_{idx}.npy'), flow_gt)

    # 绘制可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(source, cmap='hot')
    axes[0].set_title('Source')
    axes[0].axis('off')

    axes[1].imshow(target, cmap='hot')
    axes[1].set_title('Target')
    axes[1].axis('off')

    # 差异图
    diff = np.abs(target - source)
    axes[2].imshow(diff, cmap='gray')
    axes[2].set_title('Difference')
    axes[2].axis('off')

    # 变形场幅度
    flow_mag = np.sqrt(flow_gt[0] ** 2 + flow_gt[1] ** 2)
    im = axes[3].imshow(flow_mag, cmap='viridis')
    axes[3].set_title('|Flow|')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sample_{idx}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_synthetic_dataset(n_samples=100, save_dir='data/synthetic_fus',
                             image_size=(128, 100), motion_type='mixed'):
    """
    创建并保存合成数据集。

    Parameters
    ----------
    n_samples : int
        样本数量
    save_dir : str
        保存目录
    image_size : tuple
        图像尺寸
    motion_type : str
        运动类型
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    generator = SyntheticFUSGenerator(size=image_size, motion_type=motion_type)

    print(f"Generating {n_samples} synthetic fUS samples...")
    for i in range(n_samples):
        source, target, flow_gt = generator.generate_pair(seed=i)
        save_synthetic_sample(source, target, flow_gt, save_dir, idx=i)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples")

    print(f"Dataset saved to {save_dir}")


# -----------------------------------------------------------------------------
#  演示
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # 生成并显示样本
    print("Generating synthetic fUS sample...")

    generator = SyntheticFUSGenerator(size=(128, 100), motion_type='mixed')
    source, target, flow_gt = generator.generate_pair(seed=42)

    print(f"Source shape: {source.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Flow shape: {flow_gt.shape}")
    print(f"Max displacement: {np.sqrt(flow_gt[0]**2 + flow_gt[1]**2).max():.2f} pixels")

    # 保存可视化
    save_synthetic_sample(source, target, flow_gt, 'data/synthetic_demo', idx=0)
    print("Sample saved to data/synthetic_demo/")

    # 创建完整数据集
    # create_synthetic_dataset(n_samples=50, save_dir='data/synthetic_fus_train')
