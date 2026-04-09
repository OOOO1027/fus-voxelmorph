"""
fUS 数据管道使用示例和测试脚本。

演示：
1. 加载真实 fUS 数据（.npy/.mat）
2. 应用预处理和数据增强
3. 生成合成数据用于训练
4. 创建 DataLoader 进行批量训练
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (
    # Datasets
    FUSDataset, FUSPairDataset,
    SyntheticFUSGenerator, SyntheticFUSDataset,
    # Augmentation
    get_fus_transforms, Compose, RandomAffine2D, RandomIntensityNoise,
    # Utilities
    create_synthetic_dataset, save_synthetic_sample,
)


def demo_1_load_real_data():
    """示例 1: 加载真实 fUS 数据。"""
    print("=" * 60)
    print("Demo 1: 加载真实 fUS 数据")
    print("=" * 60)

    # 首先创建一些模拟的真实数据用于演示
    os.makedirs('demo_data/real', exist_ok=True)

    # 创建模拟的时间序列数据 (T, H, W)
    np.random.seed(42)
    n_frames = 20
    frames = []

    for t in range(n_frames):
        # 模拟血管图像 + 缓慢变化
        base = np.random.rand(128, 100) * 0.3
        # 添加一些"血管"
        for _ in range(5):
            x = np.random.randint(20, 80)
            y = np.random.randint(20, 108)
            base[y:y + 10, x:x + 5] += np.random.rand() * 0.7
        frames.append(base)

    frames = np.array(frames)
    np.save('demo_data/real/fus_timeseries.npy', frames)
    print(f"Created demo data: {frames.shape}")

    # 方式 1: 加载时间序列文件
    dataset = FUSDataset(
        data_path='demo_data/real/fus_timeseries.npy',
        target_size=(128, 100),
        normalize='minmax',
        log_transform=False,
        gaussian_sigma=None,
    )

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")  # (1, H, W)
    print(f"Value range: [{sample.min():.3f}, {sample.max():.3f}]")

    # 可视化几帧
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(dataset):
            frame = dataset[i][0].numpy()  # (H, W)
            ax.imshow(frame, cmap='hot')
            ax.set_title(f'Frame {i}')
            ax.axis('off')
    plt.suptitle('Real fUS Data Samples')
    plt.tight_layout()
    plt.savefig('demo_data/demo1_real_data.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to demo_data/demo1_real_data.png")
    plt.close()


def demo_2_preprocessing():
    """示例 2: 不同的预处理方法。"""
    print("\n" + "=" * 60)
    print("Demo 2: 预处理方法对比")
    print("=" * 60)

    # 创建有偏斜分布的数据
    np.random.seed(42)
    skewed_data = np.random.exponential(scale=2.0, size=(128, 100))
    skewed_data = np.clip(skewed_data, 0, 10)
    np.save('demo_data/skewed_sample.npy', skewed_data)

    methods = [
        ('None (raw)', dict(normalize=None)),
        ('Min-Max', dict(normalize='minmax')),
        ('Percentile (1-99)', dict(normalize='percentile')),
        ('Min-Max + Log', dict(normalize='minmax', log_transform=True)),
        ('Min-Max + Gaussian', dict(normalize='minmax', gaussian_sigma=1.0)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, kwargs) in enumerate(methods):
        dataset = FUSDataset(
            data_path='demo_data/skewed_sample.npy',
            target_size=(128, 100),
            **kwargs
        )
        frame = dataset[0][0].numpy()

        axes[idx].imshow(frame, cmap='hot')
        axes[idx].set_title(name)
        axes[idx].axis('off')

        print(f"{name:25s} - range: [{frame.min():.3f}, {frame.max():.3f}], "
              f"mean: {frame.mean():.3f}, std: {frame.std():.3f}")

    # 原始数据的直方图
    axes[-1].hist(skewed_data.flatten(), bins=50, color='blue', alpha=0.7)
    axes[-1].set_title('Raw Data Histogram')
    axes[-1].set_xlabel('Intensity')
    axes[-1].set_ylabel('Frequency')

    plt.suptitle('Preprocessing Methods Comparison')
    plt.tight_layout()
    plt.savefig('demo_data/demo2_preprocessing.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to demo_data/demo2_preprocessing.png")
    plt.close()


def demo_3_augmentation():
    """示例 3: 数据增强。"""
    print("\n" + "=" * 60)
    print("Demo 3: 数据增强")
    print("=" * 60)

    # 创建基础数据集
    dataset = FUSDataset(
        data_path='demo_data/real/fus_timeseries.npy',
        target_size=(128, 100),
        normalize='minmax',
        augmentation=None  # 先不增强
    )

    # 应用不同的增强
    augmentations = [
        ('Original', None),
        ('Affine (rot=10, trans=10)', RandomAffine2D(rotation=10, translation=10, p=1.0)),
        ('Noise (std=0.05)', RandomIntensityNoise(noise_std=0.05, p=1.0)),
        ('Combined', get_fus_transforms(train=True, rotation=5, translation=5, noise_std=0.03)),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for row, (name, aug) in enumerate(augmentations):
        dataset.augmentation = aug

        for col in range(4):
            frame = dataset[0][0].numpy()
            axes[row, col].imshow(frame, cmap='hot')
            if col == 0:
                axes[row, col].set_ylabel(name, fontsize=12)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        axes[row, 0].set_title(f'{name} - Sample 1')

    plt.suptitle('Data Augmentation Examples (4 samples each)')
    plt.tight_layout()
    plt.savefig('demo_data/demo3_augmentation.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to demo_data/demo3_augmentation.png")
    plt.close()


def demo_4_pair_dataset():
    """示例 4: 配对数据集用于配准训练。"""
    print("\n" + "=" * 60)
    print("Demo 4: 配对数据集")
    print("=" * 60)

    base_dataset = FUSDataset(
        data_path='demo_data/real/fus_timeseries.npy',
        target_size=(128, 100),
        normalize='minmax',
    )

    modes = ['consecutive', 'to_reference', 'sliding_window']

    fig, axes = plt.subplots(len(modes), 4, figsize=(16, 4 * len(modes)))

    for row, mode in enumerate(modes):
        pair_dataset = FUSPairDataset(
            base_dataset,
            mode=mode,
            ref_idx=0,
            window_size=3
        )

        print(f"\nMode '{mode}': {len(pair_dataset)} pairs")

        for col in range(min(4, len(pair_dataset))):
            source, target = pair_dataset[col]
            src_np = source[0].numpy()
            tgt_np = target[0].numpy()

            # 显示 source, target, 和差异
            ax = axes[row, col]
            ax.imshow(np.concatenate([src_np, tgt_np, np.abs(src_np - tgt_np)], axis=1),
                     cmap='hot')
            ax.set_title(f'{mode} - Pair {col}')
            ax.axis('off')

    plt.suptitle('Pair Dataset Modes (Source | Target | Diff)')
    plt.tight_layout()
    plt.savefig('demo_data/demo4_pairs.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to demo_data/demo4_pairs.png")
    plt.close()


def demo_5_synthetic_data():
    """示例 5: 合成数据生成。"""
    print("\n" + "=" * 60)
    print("Demo 5: 合成 fUS 数据")
    print("=" * 60)

    generator = SyntheticFUSGenerator(size=(128, 100), motion_type='mixed')

    motion_types = ['random', 'cardiac', 'breathing', 'mixed']

    fig, axes = plt.subplots(len(motion_types), 4, figsize=(16, 4 * len(motion_types)))

    for row, motion in enumerate(motion_types):
        generator.motion_type = motion
        source, target, flow_gt = generator.generate_pair(seed=42)

        # 计算变形幅度
        flow_mag = np.sqrt(flow_gt[0] ** 2 + flow_gt[1] ** 2)

        axes[row, 0].imshow(source, cmap='hot')
        axes[row, 0].set_title(f'{motion}: Source')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(target, cmap='hot')
        axes[row, 1].set_title(f'{motion}: Target')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(np.abs(target - source), cmap='gray')
        axes[row, 2].set_title(f'{motion}: Diff')
        axes[row, 2].axis('off')

        im = axes[row, 3].imshow(flow_mag, cmap='viridis')
        axes[row, 3].set_title(f'{motion}: |Flow| (max={flow_mag.max():.1f}px)')
        axes[row, 3].axis('off')
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046)

    plt.suptitle('Synthetic fUS Data with Different Motion Types')
    plt.tight_layout()
    plt.savefig('demo_data/demo5_synthetic.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to demo_data/demo5_synthetic.png")
    plt.close()

    # 保存样本
    save_synthetic_sample(source, target, flow_gt, 'demo_data/synthetic', idx=0)
    print("Saved synthetic sample to demo_data/synthetic/")


def demo_6_dataloader():
    """示例 6: 使用 DataLoader 进行批量训练。"""
    print("\n" + "=" * 60)
    print("Demo 6: DataLoader 批量训练")
    print("=" * 60)

    # 使用合成数据演示
    dataset = SyntheticFUSDataset(
        size=100,
        image_size=(128, 100),
        motion_type='mixed',
        max_displacement=8,
        noise_level=0.05
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # 合成数据不需要多进程
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # 获取一个 batch
    batch = next(iter(dataloader))
    sources, targets, flows_gt = batch

    print(f"\nBatch shapes:")
    print(f"  sources: {sources.shape}")  # (B, 1, H, W)
    print(f"  targets: {targets.shape}")
    print(f"  flows_gt: {flows_gt.shape}")  # (B, 2, H, W)

    # 可视化 batch
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i in range(4):
        src = sources[i, 0].numpy()
        tgt = targets[i, 0].numpy()
        flow = flows_gt[i].numpy()
        flow_mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2)

        axes[i, 0].imshow(src, cmap='hot')
        axes[i, 0].set_title(f'Batch {i}: Source')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(tgt, cmap='hot')
        axes[i, 1].set_title(f'Batch {i}: Target')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(np.abs(tgt - src), cmap='gray')
        axes[i, 2].set_title(f'Batch {i}: Diff')
        axes[i, 2].axis('off')

        im = axes[i, 3].imshow(flow_mag, cmap='viridis')
        axes[i, 3].set_title(f'Batch {i}: |Flow|')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046)

    plt.suptitle('DataLoader Batch Samples')
    plt.tight_layout()
    plt.savefig('demo_data/demo6_dataloader.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to demo_data/demo6_dataloader.png")
    plt.close()

    # 模拟训练循环
    print("\nSimulating training loop...")
    for epoch in range(2):
        total_loss = 0
        for batch_idx, (src, tgt, flow) in enumerate(dataloader):
            # 这里可以放入模型训练代码
            # loss = model(src, tgt, flow)
            dummy_loss = torch.rand(1).item()
            total_loss += dummy_loss

            if batch_idx >= 4:  # 只跑几个 batch
                break

        print(f"Epoch {epoch + 1}: avg loss = {total_loss / 5:.4f}")


def demo_7_matlab_loading():
    """示例 7: 加载 MATLAB (.mat) 文件。"""
    print("\n" + "=" * 60)
    print("Demo 7: MATLAB 文件加载")
    print("=" * 60)

    try:
        from scipy.io import savemat

        # 创建模拟的 .mat 文件
        os.makedirs('demo_data/matlab', exist_ok=True)

        data = np.random.rand(20, 128, 100).astype(np.float32)
        savemat('demo_data/matlab/fus_data.mat', {'data': data})
        print("Created demo MATLAB file: demo_data/matlab/fus_data.mat")

        # 加载
        dataset = FUSDataset(
            data_path='demo_data/matlab/fus_data.mat',
            target_size=(128, 100),
            normalize='minmax',
            mat_key='data'
        )

        print(f"Loaded {len(dataset)} frames from .mat file")
        sample = dataset[0]
        print(f"Sample shape: {sample.shape}")

        # 可视化
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, ax in enumerate(axes):
            frame = dataset[i][0].numpy()
            ax.imshow(frame, cmap='hot')
            ax.set_title(f'MATLAB Frame {i}')
            ax.axis('off')

        plt.suptitle('MATLAB Data Loading')
        plt.tight_layout()
        plt.savefig('demo_data/demo7_matlab.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to demo_data/demo7_matlab.png")
        plt.close()

    except ImportError:
        print("scipy not installed, skipping MATLAB demo")
        print("Install with: pip install scipy")


def run_all_demos():
    """运行所有演示。"""
    print("\n" + "=" * 60)
    print("fUS Data Pipeline Demo")
    print("=" * 60)

    demo_1_load_real_data()
    demo_2_preprocessing()
    demo_3_augmentation()
    demo_4_pair_dataset()
    demo_5_synthetic_data()
    demo_6_dataloader()
    demo_7_matlab_loading()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("Check demo_data/ directory for visualizations")
    print("=" * 60)


if __name__ == '__main__':
    run_all_demos()
