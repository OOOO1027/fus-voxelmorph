"""
Dataset classes for functional ultrasound (fUS) image registration.

fUS Power Doppler 图像特点：
- 格式：.mat (MATLAB) 或 .npy/.npz (NumPy)
- 尺寸：128 x 100 像素（12.8mm x 10mm FOV, 100μm 分辨率）
- 强度：Power Doppler 信号，值域为正数，分布偏斜
- 血管结构：亮的管状/点状特征

数据预处理：
1. 归一化到 [0, 1]（min-max 或 percentile-based）
2. 可选：log 变换以压缩动态范围
3. 可选：空间滤波（高斯平滑）
4. Resize/pad 到统一尺寸
"""

import os
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


# -----------------------------------------------------------------------------
#  预处理函数
# -----------------------------------------------------------------------------

def normalize_frame(frame, method='minmax', percentile=None):
    """
    归一化 fUS 帧到 [0, 1]。

    Parameters
    ----------
    frame : np.ndarray
        输入帧，形状 (H, W) 或 (C, H, W)
    method : str
        'minmax' - 简单的 min-max 归一化
        'percentile' - 基于 percentile 的鲁棒归一化
    percentile : tuple, optional
        (low, high) percentile，默认 (1, 99)

    Returns
    -------
    np.ndarray : 归一化后的帧 [0, 1]
    """
    frame = frame.astype(np.float32)

    if method == 'minmax':
        fmin, fmax = frame.min(), frame.max()
        if fmax - fmin > 1e-8:
            frame = (frame - fmin) / (fmax - fmin)
        else:
            frame = np.zeros_like(frame)

    elif method == 'percentile':
        if percentile is None:
            percentile = (1, 99)
        p_low, p_high = np.percentile(frame, percentile)
        frame = np.clip(frame, p_low, p_high)
        if p_high - p_low > 1e-8:
            frame = (frame - p_low) / (p_high - p_low)
        else:
            frame = np.zeros_like(frame)

    return frame


def log_transform(frame, epsilon=1e-6):
    """
    Log 变换压缩动态范围。
    
    公式: log(1 + x / epsilon) / log(1 + 1/epsilon)
    将输出归一化到 [0, 1]
    """
    frame = np.log1p(frame / epsilon)
    frame = frame / np.log1p(1.0 / epsilon)
    return np.clip(frame, 0, 1)


def gaussian_smooth(frame, sigma=1.0):
    """高斯平滑滤波。"""
    from scipy.ndimage import gaussian_filter
    if frame.ndim == 2:
        return gaussian_filter(frame, sigma=sigma)
    else:
        return gaussian_filter(frame, sigma=(0, sigma, sigma))


def resize_or_pad(frame, target_size=(128, 100), mode='bilinear'):
    """
    Resize 或 pad 帧到目标尺寸。

    Parameters
    ----------
    frame : np.ndarray or torch.Tensor
        输入帧 (H, W) 或 (C, H, W) 或 (B, C, H, W)
    target_size : tuple
        目标尺寸 (H, W)，默认 (128, 100)
    mode : str
        插值模式: 'nearest', 'bilinear', 'bicubic'

    Returns
    -------
    torch.Tensor : 调整后的帧
    """
    is_numpy = isinstance(frame, np.ndarray)
    if is_numpy:
        frame = torch.from_numpy(frame).float()

    # 确保是 4D tensor (B, C, H, W)
    orig_ndim = frame.ndim
    if orig_ndim == 2:
        frame = frame.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif orig_ndim == 3:
        frame = frame.unsqueeze(0)  # (1, C, H, W)

    _, C, H, W = frame.shape
    target_H, target_W = target_size

    if H == target_H and W == target_W:
        output = frame
    elif H <= target_H and W <= target_W:
        # Pad
        pad_h = target_H - H
        pad_w = target_W - W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        output = F.pad(frame, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    else:
        # Resize
        output = F.interpolate(frame, size=target_size, mode=mode, align_corners=False)

    # 恢复原始维度
    if orig_ndim == 2:
        output = output[0, 0]
    elif orig_ndim == 3:
        output = output[0]

    if is_numpy:
        output = output.numpy()

    return output


# -----------------------------------------------------------------------------
#  数据增强
# -----------------------------------------------------------------------------

class RandomAffine2D:
    """
    随机 2D 仿射变换（旋转、平移、缩放）。

    用于模拟跨 session 的微小差异或轻微运动。

    Parameters
    ----------
    rotation : float
        最大旋转角度（度），默认 5
    translation : float or tuple
        最大平移像素数，默认 5
    scale : tuple
        缩放范围 (min, max)，默认 (0.95, 1.05)
    p : float
        应用变换的概率，默认 0.5
    """

    def __init__(self, rotation=5, translation=5, scale=(0.95, 1.05), p=0.5):
        self.rotation = rotation
        self.translation = translation if isinstance(translation, tuple) else (translation, translation)
        self.scale = scale
        self.p = p

    def __call__(self, frame):
        if np.random.random() > self.p:
            return frame

        is_numpy = isinstance(frame, np.ndarray)
        if is_numpy:
            frame = torch.from_numpy(frame).float()

        # 确保是 3D (C, H, W)
        if frame.ndim == 2:
            frame = frame.unsqueeze(0)

        C, H, W = frame.shape

        # 随机参数
        angle = np.random.uniform(-self.rotation, self.rotation)
        tx = np.random.uniform(-self.translation[0], self.translation[0])
        ty = np.random.uniform(-self.translation[1], self.translation[1])
        scale = np.random.uniform(self.scale[0], self.scale[1])

        # 构建仿射矩阵
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # 旋转+缩放矩阵
        theta = torch.tensor([
            [cos_a * scale, -sin_a * scale, tx * 2 / W],
            [sin_a * scale, cos_a * scale, ty * 2 / H]
        ], dtype=torch.float32)

        # 应用变换
        grid = F.affine_grid(theta.unsqueeze(0), frame.unsqueeze(0).size(), align_corners=False)
        frame = F.grid_sample(frame.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=False)[0]

        if is_numpy:
            frame = frame.numpy()

        return frame


class RandomIntensityNoise:
    """
    随机强度扰动（加性高斯噪声 + 乘性噪声 + 亮度调整）。

    Parameters
    ----------
    noise_std : float
        加性噪声标准差，默认 0.02
    multiplicative_range : tuple
        乘性噪声范围 (1 - range, 1 + range)，默认 (0.95, 1.05)
    brightness_range : tuple
        亮度调整范围，默认 (-0.05, 0.05)
    p : float
        应用变换的概率，默认 0.5
    """

    def __init__(self, noise_std=0.02, multiplicative_range=(0.95, 1.05),
                 brightness_range=(-0.05, 0.05), p=0.5):
        self.noise_std = noise_std
        self.multiplicative_range = multiplicative_range
        self.brightness_range = brightness_range
        self.p = p

    def __call__(self, frame):
        if np.random.random() > self.p:
            return frame

        is_numpy = isinstance(frame, np.ndarray)
        if is_numpy:
            frame = torch.from_numpy(frame).float()

        # 加性噪声
        if self.noise_std > 0:
            noise = torch.randn_like(frame) * self.noise_std
            frame = frame + noise

        # 乘性噪声
        if self.multiplicative_range is not None:
            mult = np.random.uniform(self.multiplicative_range[0], self.multiplicative_range[1])
            frame = frame * mult

        # 亮度调整
        if self.brightness_range is not None:
            brightness = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
            frame = frame + brightness

        frame = torch.clamp(frame, 0, 1)

        if is_numpy:
            frame = frame.numpy()

        return frame


class RandomCrop:
    """
    随机裁剪到指定尺寸，然后 resize 回原始尺寸（模拟局部变形）。

    Parameters
    ----------
    crop_scale : tuple
        裁剪比例范围 (min, max)，默认 (0.9, 1.0)
    p : float
        应用裁剪的概率，默认 0.3
    """

    def __init__(self, crop_scale=(0.9, 1.0), p=0.3):
        self.crop_scale = crop_scale
        self.p = p

    def __call__(self, frame):
        if np.random.random() > self.p:
            return frame

        is_numpy = isinstance(frame, np.ndarray)
        if is_numpy:
            frame = torch.from_numpy(frame).float()

        if frame.ndim == 2:
            C, H, W = 1, frame.shape[0], frame.shape[1]
            frame = frame.unsqueeze(0)
        else:
            C, H, W = frame.shape

        scale = np.random.uniform(self.crop_scale[0], self.crop_scale[1])
        new_H, new_W = int(H * scale), int(W * scale)

        # 随机起始位置
        top = np.random.randint(0, H - new_H + 1)
        left = np.random.randint(0, W - new_W + 1)

        # 裁剪并 resize 回去
        cropped = frame[:, top:top + new_H, left:left + new_W]
        frame = F.interpolate(cropped.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)[0]

        if is_numpy:
            frame = frame.numpy()

        return frame


class RandomElasticDeformation:
    """
    随机弹性变形（B-spline 风格），模拟跨 session 的非刚体形变。

    在低分辨率网格上生成随机位移，再上采样到原始尺寸，
    产生平滑的非刚体变形场。这是对 VoxelMorph 训练数据
    最关键的增广——因为模型本身就要学习非刚体形变。

    Parameters
    ----------
    grid_size : int
        控制点网格的边长（越小越平滑），默认 5
    magnitude : float
        最大位移幅度（像素），默认 4.0
    p : float
        应用变换的概率，默认 0.5
    """

    def __init__(self, grid_size=5, magnitude=4.0, p=0.5):
        self.grid_size = grid_size
        self.magnitude = magnitude
        self.p = p

    def __call__(self, frame):
        if np.random.random() > self.p:
            return frame

        is_numpy = isinstance(frame, np.ndarray)
        if is_numpy:
            frame = torch.from_numpy(frame).float()

        if frame.ndim == 2:
            frame = frame.unsqueeze(0)

        C, H, W = frame.shape

        # 在低分辨率网格上生成随机位移
        dx = np.random.uniform(-self.magnitude, self.magnitude,
                               (1, 1, self.grid_size, self.grid_size)).astype(np.float32)
        dy = np.random.uniform(-self.magnitude, self.magnitude,
                               (1, 1, self.grid_size, self.grid_size)).astype(np.float32)

        # 上采样到原始尺寸（产生平滑的变形场）
        dx = F.interpolate(torch.from_numpy(dx), size=(H, W),
                           mode='bicubic', align_corners=False)[0, 0]
        dy = F.interpolate(torch.from_numpy(dy), size=(H, W),
                           mode='bicubic', align_corners=False)[0, 0]

        # 构建采样网格：identity + displacement
        grid_y = torch.linspace(-1, 1, H)
        grid_x = torch.linspace(-1, 1, W)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing='ij')

        # 将像素位移转换为归一化坐标位移
        sample_grid = torch.stack([
            gx + dx * 2.0 / (W - 1),
            gy + dy * 2.0 / (H - 1),
        ], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        frame = F.grid_sample(
            frame.unsqueeze(0), sample_grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )[0]

        if is_numpy:
            frame = frame.numpy()

        return frame


class RandomFlip:
    """
    随机水平/垂直翻转。

    Parameters
    ----------
    horizontal : bool
        是否水平翻转，默认 True
    vertical : bool
        是否垂直翻转，默认 False
    p : float
        每种翻转的独立概率，默认 0.5
    """

    def __init__(self, horizontal=True, vertical=False, p=0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(self, frame):
        is_numpy = isinstance(frame, np.ndarray)
        if is_numpy:
            frame = torch.from_numpy(frame).float()

        if self.horizontal and np.random.random() < self.p:
            frame = torch.flip(frame, dims=[-1])
        if self.vertical and np.random.random() < self.p:
            frame = torch.flip(frame, dims=[-2])

        if is_numpy:
            frame = frame.numpy()
        return frame


class Compose:
    """组合多个变换。"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame):
        for t in self.transforms:
            frame = t(frame)
        return frame


class PairedCompose:
    """
    配对增强：几何变换对两张图使用相同随机状态，强度变换独立施加。

    Parameters
    ----------
    geometric_transforms : list
        几何变换列表 (RandomAffine2D, RandomElasticDeformation, RandomCrop, RandomFlip)
        对 moving 和 fixed 使用相同随机种子，保持空间对应关系。
    intensity_transforms : list
        强度变换列表 (RandomIntensityNoise 等)
        对 moving 和 fixed 独立施加，模拟不同成像条件。
    """

    def __init__(self, geometric_transforms=None, intensity_transforms=None):
        self.geometric_transforms = geometric_transforms or []
        self.intensity_transforms = intensity_transforms or []

    def __call__(self, moving, fixed, mask=None):
        # 几何变换：使用相同随机种子确保一致性
        seed = np.random.randint(0, 2**31)
        for t in self.geometric_transforms:
            np.random.seed(seed)
            torch.manual_seed(seed)
            moving = t(moving)
            np.random.seed(seed)
            torch.manual_seed(seed)
            fixed = t(fixed)
            if mask is not None:
                np.random.seed(seed)
                torch.manual_seed(seed)
                mask = t(mask)

        # 强度变换：独立施加（模拟不同 session 的成像差异）
        for t in self.intensity_transforms:
            moving = t(moving)
            fixed = t(fixed)

        return moving, fixed, mask


class RandomGammaCorrection:
    """
    随机 Gamma 校正，模拟不同 Power Doppler 增益设置。

    Parameters
    ----------
    gamma_range : tuple
        Gamma 范围 (min, max)，默认 (0.7, 1.3)
    p : float
        应用概率
    """

    def __init__(self, gamma_range=(0.7, 1.3), p=0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, frame):
        if np.random.random() > self.p:
            return frame

        is_numpy = isinstance(frame, np.ndarray)
        if is_numpy:
            frame = torch.from_numpy(frame).float()

        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        frame = torch.clamp(frame, 0, 1)
        frame = torch.pow(frame + 1e-8, gamma)
        frame = torch.clamp(frame, 0, 1)

        if is_numpy:
            frame = frame.numpy()
        return frame


# -----------------------------------------------------------------------------
#  FUS Dataset
# -----------------------------------------------------------------------------

class FUSDataset(Dataset):
    """
    fUS Power Doppler 数据集。

    支持格式：
    - .npy / .npz: NumPy 数组格式
    - .mat: MATLAB 格式（需要 scipy）

    支持模式：
    - 'directory': 文件夹中的多个 .npy/.mat 文件
    - 'timeseries': 单个时间序列文件 (T, H, W) 或 (T, C, H, W)
    - 'filelist': 显式文件列表

    Parameters
    ----------
    data_path : str
        数据路径（文件夹或文件）
    target_size : tuple
        目标尺寸 (H, W)，默认 (128, 100)
    normalize : str or None
        归一化方法: 'minmax', 'percentile', None
    log_transform : bool
        是否应用 log 变换，默认 False
    gaussian_sigma : float or None
        高斯平滑 sigma，默认 None（不应用）
    mat_key : str
        从 .mat 文件加载时使用的变量名，默认 'data'
    augmentation : Compose or None
        数据增强变换（仅用于训练）
    """

    def __init__(self, data_path, target_size=(128, 100), normalize='minmax',
                 log_transform=False, gaussian_sigma=None, mat_key='data',
                 augmentation=None):
        self.data_path = Path(data_path)
        self.target_size = target_size
        self.normalize = normalize
        self.log_transform = log_transform
        self.gaussian_sigma = gaussian_sigma
        self.mat_key = mat_key
        self.augmentation = augmentation

        # 确定数据模式
        if self.data_path.is_dir():
            self.mode = 'directory'
            self.files = []
            for ext in ['*.npy', '*.npz', '*.mat']:
                self.files.extend(sorted(self.data_path.glob(ext)))
            if len(self.files) == 0:
                raise ValueError(f"No .npy/.npz/.mat files found in {data_path}")
        elif self.data_path.is_file():
            self.mode = 'timeseries'
            self.data = self._load_file(self.data_path)
        else:
            raise ValueError(f"data_path must be a directory or file, got {data_path}")

    def _load_file(self, filepath):
        """加载单个文件。"""
        filepath = Path(filepath)

        if filepath.suffix == '.npy':
            data = np.load(str(filepath))
        elif filepath.suffix == '.npz':
            # 尝试加载第一个数组
            loaded = np.load(str(filepath))
            if len(loaded.files) == 1:
                data = loaded[loaded.files[0]]
            else:
                # 如果有 'data' 键，使用它
                data = loaded.get('data', loaded[loaded.files[0]])
        elif filepath.suffix == '.mat':
            try:
                from scipy.io import loadmat
                mat = loadmat(str(filepath))
                data = mat.get(self.mat_key, mat.get('data', list(mat.values())[-1]))
            except ImportError:
                raise ImportError("scipy is required to load .mat files. Install with: pip install scipy")
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        return data

    def _preprocess(self, frame):
        """预处理单帧。"""
        # 确保是 2D
        if frame.ndim == 3 and frame.shape[0] == 1:
            frame = frame[0]  # (1, H, W) -> (H, W)
        elif frame.ndim == 3:
            # 多通道，取第一个或平均
            frame = frame[0]

        # Log 变换（在归一化前）
        if self.log_transform:
            frame = log_transform(frame)

        # 归一化
        if self.normalize:
            frame = normalize_frame(frame, method=self.normalize)

        # 高斯平滑
        if self.gaussian_sigma is not None and self.gaussian_sigma > 0:
            frame = gaussian_smooth(frame, sigma=self.gaussian_sigma)

        # Resize/pad 到目标尺寸
        if self.target_size is not None:
            frame = resize_or_pad(frame, target_size=self.target_size)

        # 添加通道维度: (H, W) -> (1, H, W)
        if frame.ndim == 2:
            frame = frame[np.newaxis]

        return frame.astype(np.float32)

    def __len__(self):
        if self.mode == 'directory':
            return len(self.files)
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.mode == 'directory':
            frame = self._load_file(self.files[idx])
            # 如果是多帧文件，取第一帧
            if frame.ndim == 3 and frame.shape[0] > 1:
                frame = frame[0]
        else:
            frame = self.data[idx]

        # 预处理
        frame = self._preprocess(frame)

        # 数据增强（仅训练时）
        if self.augmentation is not None:
            frame = self.augmentation(frame)

        return torch.from_numpy(frame)

    def get_frame_info(self, idx):
        """获取帧的元信息。"""
        if self.mode == 'directory':
            return {'file': str(self.files[idx]), 'index': 0}
        else:
            return {'file': str(self.data_path), 'index': idx}


# -----------------------------------------------------------------------------
#  Pair Dataset for Registration
# -----------------------------------------------------------------------------

class FUSPairDataset(Dataset):
    """
    生成 (source, target) 对用于配准训练/评估。

    Parameters
    ----------
    base_dataset : FUSDataset
        基础数据集
    mode : str
        配对策略: 'consecutive', 'random', 'to_reference', 'sliding_window'
    ref_idx : int
        参考帧索引（用于 'to_reference' 模式）
    window_size : int
        滑动窗口大小（用于 'sliding_window' 模式）
    max_pairs : int or None
        最大配对数量，默认 None（使用全部）
    """

    def __init__(self, base_dataset, mode='consecutive', ref_idx=0,
                 window_size=5, max_pairs=None):
        self.base = base_dataset
        self.mode = mode
        self.ref_idx = ref_idx
        self.window_size = window_size
        self.max_pairs = max_pairs

        self._build_pairs()

    def _build_pairs(self):
        """构建配对索引列表。"""
        n = len(self.base)
        self.pairs = []

        if self.mode == 'consecutive':
            for i in range(n - 1):
                self.pairs.append((i, i + 1))

        elif self.mode == 'to_reference':
            for i in range(n):
                if i != self.ref_idx:
                    self.pairs.append((i, self.ref_idx))

        elif self.mode == 'random':
            # 随机配对，数量等于数据集大小
            for i in range(n):
                j = np.random.randint(0, n)
                self.pairs.append((i, j))

        elif self.mode == 'sliding_window':
            # 滑动窗口内配对
            for i in range(n):
                for j in range(1, self.window_size + 1):
                    if i + j < n:
                        self.pairs.append((i, i + j))

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # 限制最大配对数
        if self.max_pairs is not None and len(self.pairs) > self.max_pairs:
            indices = np.random.choice(len(self.pairs), self.max_pairs, replace=False)
            self.pairs = [self.pairs[i] for i in sorted(indices)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_idx, tgt_idx = self.pairs[idx]
        source = self.base[src_idx]
        target = self.base[tgt_idx]
        return source, target

    def get_pair_info(self, idx):
        """获取配对的元信息。"""
        src_idx, tgt_idx = self.pairs[idx]
        return {
            'source': self.base.get_frame_info(src_idx),
            'target': self.base.get_frame_info(tgt_idx)
        }


# -----------------------------------------------------------------------------
#  工具函数
# -----------------------------------------------------------------------------

def get_fus_transforms(train=True, rotation=5, translation=5,
                       noise_std=0.02, crop_scale=(0.9, 1.0),
                       elastic=True, elastic_grid=5, elastic_magnitude=4.0,
                       flip_horizontal=True):
    """
    获取标准 fUS 数据增强组合。

    Parameters
    ----------
    train : bool
        是否训练模式（启用增强）
    rotation : float
        最大旋转角度
    translation : float
        最大平移像素
    noise_std : float
        噪声标准差
    crop_scale : tuple
        裁剪比例
    elastic : bool
        是否启用弹性变形增广
    elastic_grid : int
        弹性变形控制点网格大小
    elastic_magnitude : float
        弹性变形最大位移（像素）
    flip_horizontal : bool
        是否启用水平翻转

    Returns
    -------
    Compose or None
    """
    if not train:
        return None

    transforms = [
        RandomAffine2D(rotation=rotation, translation=translation, p=0.5),
        RandomIntensityNoise(noise_std=noise_std, p=0.5),
        RandomCrop(crop_scale=crop_scale, p=0.3),
    ]

    if elastic:
        transforms.append(RandomElasticDeformation(
            grid_size=elastic_grid, magnitude=elastic_magnitude, p=0.5
        ))

    if flip_horizontal:
        transforms.append(RandomFlip(horizontal=True, vertical=False, p=0.5))

    return Compose(transforms)
