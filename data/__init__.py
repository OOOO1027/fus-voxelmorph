"""
fUS 数据加载模块。

包含：
- FUSDataset: 真实 fUS 数据加载（支持 .npy/.npz/.mat）
- FUSPairDataset: 配准配对数据集
- CrossSessionPairDataset: 跨 session 预定义配对数据集（DL-Prep 产物）
- SyntheticFUSGenerator: 合成数据生成器
- 预处理函数和增强变换
"""

from .fus_dataset import (
    # Datasets
    FUSDataset,
    FUSPairDataset,
    # Preprocessing functions
    normalize_frame,
    log_transform,
    gaussian_smooth,
    resize_or_pad,
    # Augmentation
    RandomAffine2D,
    RandomIntensityNoise,
    RandomCrop,
    RandomElasticDeformation,
    RandomFlip,
    RandomGammaCorrection,
    Compose,
    PairedCompose,
    get_fus_transforms,
)

from .cross_session_dataset import (
    CrossSessionPairDataset,
    CrossSessionCollator,
)

from .synthetic_fus import (
    # Generators
    SyntheticFUSGenerator,
    SyntheticFUSDataset,
    # Vessel generation
    generate_vessel_tree,
    generate_vessel_network,
    # Deformation generation
    generate_deformation_field,
    generate_cardiac_like_motion,
    generate_breathing_motion,
    # Utilities
    apply_deformation,
    save_synthetic_sample,
    create_synthetic_dataset,
)

__all__ = [
    # Datasets
    'FUSDataset',
    'FUSPairDataset',
    'CrossSessionPairDataset',
    'CrossSessionCollator',
    'SyntheticFUSGenerator',
    'SyntheticFUSDataset',
    # Preprocessing
    'normalize_frame',
    'log_transform',
    'gaussian_smooth',
    'resize_or_pad',
    # Augmentation
    'RandomAffine2D',
    'RandomIntensityNoise',
    'RandomCrop',
    'RandomElasticDeformation',
    'RandomFlip',
    'RandomGammaCorrection',
    'Compose',
    'PairedCompose',
    'get_fus_transforms',
    # Synthetic generation
    'generate_vessel_tree',
    'generate_vessel_network',
    'generate_deformation_field',
    'generate_cardiac_like_motion',
    'generate_breathing_motion',
    'apply_deformation',
    'save_synthetic_sample',
    'create_synthetic_dataset',
]
