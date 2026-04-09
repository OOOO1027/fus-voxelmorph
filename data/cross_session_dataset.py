"""
Cross-session pair dataset for fUS image registration.

直接��� DL-Prep bundle 的 .npz 训练资产加载预定义的跨 session
(moving, fixed, mask) ��，用于训练和评估。

每个 .npz 文件包含:
    - fixed_raw:     (H, W) float32 — 目标图像 (current session neurovascular_map)
    - moving_raw:    (H, W) float32 — 源图像 (previous session neurovascular_map)
    - moving_rigid:  (H, W) float32 — Stage 3B 刚体预对齐后的源图像 (Mode B 输入)
    - valid_mask:    (H, W) float32 — 有效区域掩码 (1=真实图像, 0=padding)
    - pair_id:       str — 配对标识符
    - canvas_shape_hw, native_shape_hw, pad_* — 尺寸与填���元数据

支持两种训练模式:
    - Mode A: moving_raw + fixed_raw (原���输入)
    - Mode B: moving_rigid + fixed_raw (刚体预对齐, 推荐)
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .fus_dataset import normalize_frame, log_transform as apply_log_transform


class CrossSessionPairDataset(Dataset):
    """
    加载 DL-Prep ���结的跨 session 配对数据 (.npz 格式)。

    Parameters
    ----------
    data_dir : str
        DL-Prep bundle 根目录 (包含 dl_prep/ 子目录)
    split : str
        数据划分: 'train', 'val', 'test', 'all'
    mode : str
        训练模式: 'A' (raw input) 或 'B' (rigid-prealigned, 推荐)
    use_padded : bool
        是否使用 padded_canonical 资产 (推荐 True)
    normalize : str or None
        归一化方法: 'minmax', 'percentile', None
    percentile : tuple
        percentile 归一化参数, 默认 (1, 99)
    apply_log : bool
        是否应用 log 变换 (压缩 Power Doppler 动态范围)
    augmentation : callable or None
        数据增强变换 (仅���练时)
    """

    def __init__(self, data_dir, split='train', mode='B', use_padded=True,
                 normalize='percentile', percentile=(1, 99), apply_log=True,
                 augmentation=None, include_reverse=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.mode = mode.upper()
        self.use_padded = use_padded
        self.normalize = normalize
        self.percentile = percentile
        self.apply_log = apply_log
        self.augmentation = augmentation
        self.include_reverse = include_reverse

        assert self.mode in ('A', 'B'), f"mode must be 'A' or 'B', got '{mode}'"

        # 自动定位 bundle ���构
        self._locate_paths()

        # 加载配对列表
        self._load_pairs()

        # 加载 baseline reference (Stage 3B NCC 等)
        self._load_baseline_reference()

    def _locate_paths(self):
        """定位 bundle 内部的关���路径。"""
        # 检查��否直接指向 bundle 根目录
        dl_prep = self.data_dir / 'dl_prep'
        if not dl_prep.exists():
            # 可能指向包含 bundle 的父目录，搜索子目录
            for child in self.data_dir.iterdir():
                if child.is_dir() and (child / 'dl_prep').exists():
                    dl_prep = child / 'dl_prep'
                    break

        self.dl_prep_dir = dl_prep

        # NPZ 资产目录
        asset_subdir = 'padded_canonical' if self.use_padded else 'native'
        self.npz_dir = dl_prep / 'training_assets' / 'npz' / asset_subdir

        # 配对表与 split 目录
        self.manifests_dir = dl_prep / 'manifests'
        self.splits_dir = dl_prep / 'splits'
        self.eval_dir = dl_prep / 'eval'

    def _load_pairs(self):
        """加载配对列表并按 split 过滤。"""
        # 优先��� split CSV 加载
        if self.split != 'all':
            split_csv = self.splits_dir / f'{self.split}_pairs.csv'
            if split_csv.exists():
                self.pairs_df = pd.read_csv(split_csv)
                self.pair_ids = self.pairs_df['pair_id'].tolist()
                return

        # 从主 pair_dataset.csv 加载
        csv_path = self.manifests_dir / 'pair_dataset.csv'
        if csv_path.exists():
            self.pairs_df = pd.read_csv(csv_path)
            if self.split != 'all':
                self.pairs_df = self.pairs_df[
                    self.pairs_df['split'] == self.split
                ].reset_index(drop=True)
            self.pair_ids = self.pairs_df['pair_id'].tolist()
            return

        # 从 npz 文件名自动发现
        self.pairs_df = None
        self.pair_ids = []
        if self.npz_dir.exists():
            for f in sorted(self.npz_dir.glob('*.npz')):
                self.pair_ids.append(f.stem)

    def _load_baseline_reference(self):
        """加载 Stage 3B baseline NCC 作为对比参考。"""
        self.baseline_ref = {}
        ref_csv = self.eval_dir / 'baseline_reference.csv'
        if ref_csv.exists():
            df = pd.read_csv(ref_csv)
            for _, row in df.iterrows():
                self.baseline_ref[row['pair_id']] = {
                    'pre_ncc': row.get('pre_mean_ncc'),
                    'pre_edge_ncc': row.get('pre_mean_edge_ncc'),
                    'stage3b_ncc': row.get('stage3b_backbone_mean_ncc'),
                    'stage3b_edge_ncc': row.get('stage3b_backbone_mean_edge_ncc'),
                }

    def _preprocess(self, data):
        """预处理单个图像。"""
        data = data.astype(np.float32)

        # 先 percentile 归一化（在原始值域上展开，避免 log 后值域坍缩）
        if self.normalize:
            data = normalize_frame(data, method=self.normalize,
                                   percentile=self.percentile)

        # 再 log 变换（在 [0,1] 范围上压缩偏斜分布）
        if self.apply_log:
            data = apply_log_transform(data)

        return data

    def __len__(self):
        n = len(self.pair_ids)
        return n * 2 if self.include_reverse else n

    def __getitem__(self, idx):
        # 反向配对支持：idx >= len(pair_ids) 时交换 moving 和 fixed
        n_original = len(self.pair_ids)
        is_reversed = self.include_reverse and idx >= n_original
        real_idx = idx - n_original if is_reversed else idx

        pair_id = self.pair_ids[real_idx]
        npz_path = self.npz_dir / f'{pair_id}.npz'

        npz = np.load(str(npz_path), allow_pickle=True)

        # 选择 moving 图像
        if self.mode == 'B' and 'moving_rigid' in npz:
            moving = npz['moving_rigid']
        else:
            moving = npz['moving_raw']

        fixed = npz['fixed_raw']
        mask = npz['valid_mask']

        # 反向配对：交换 moving 和 fixed
        if is_reversed:
            moving, fixed = fixed, moving

        # 预处理
        moving = self._preprocess(moving)
        fixed = self._preprocess(fixed)

        # 添加临时通道维度
        if moving.ndim == 2:
            moving = moving[np.newaxis]
        if fixed.ndim == 2:
            fixed = fixed[np.newaxis]
        if mask.ndim == 2:
            mask = mask[np.newaxis]

        # 数据增强
        if self.augmentation is not None:
            from .fus_dataset import PairedCompose
            if isinstance(self.augmentation, PairedCompose):
                # 配对增强：几何变换一致，强度变换独立
                moving, fixed, mask = self.augmentation(moving, fixed, mask)
            else:
                # 向后兼容：旧版 Compose 仅对 moving 增强（不推荐）
                moving = self.augmentation(moving)

        # 转为 tensor
        moving = torch.from_numpy(np.ascontiguousarray(moving)).float()
        fixed = torch.from_numpy(np.ascontiguousarray(fixed)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).float()

        return {
            'moving': moving,
            'fixed': fixed,
            'mask': mask,
            'pair_id': f'{pair_id}__rev' if is_reversed else pair_id,
        }

    def get_baseline_ncc(self, pair_id):
        """获取某对的 Stage 3B baseline NCC (用于对比)。"""
        return self.baseline_ref.get(pair_id, {})

    def get_all_baseline_ncc(self):
        """获取当前 split 所有对的 baseline NCC。"""
        result = {}
        for pid in self.pair_ids:
            ref = self.baseline_ref.get(pid, {})
            if ref:
                result[pid] = ref
        return result


class CrossSessionCollator:
    """
    自定义 collate 函数, ���理 dict 格式的 batch。
    """

    def __call__(self, batch):
        return {
            'moving': torch.stack([b['moving'] for b in batch]),
            'fixed': torch.stack([b['fixed'] for b in batch]),
            'mask': torch.stack([b['mask'] for b in batch]),
            'pair_id': [b['pair_id'] for b in batch],
        }
