"""
Training script for fUS-VoxelMorph 2D registration (with new data pipeline).

支持两种数据模式：
1. 时间序列帧（原有模式）: 从文件夹/时序文件自动生成配对
2. 跨 session 预定义配对（新模式）: 从 DL-Prep 冻结的 pair_dataset.csv 加载

Usage:
    # 跨 session 模式（推荐）
    python train_v2.py --config configs/cross_session.yaml

    # 原有时间序列模式
    python train_v2.py --config configs/data_pipeline.yaml

    # 合成数据模式
    python train_v2.py --config configs/data_pipeline.yaml --synthetic
"""

import argparse
import glob
import os
import time

import yaml
import torch
from torch.utils.data import DataLoader, random_split

from models import VxmDense2D
from losses import NCC, MultiScaleNCC, MSE, Grad, Diffusion, BendingEnergy, RegistrationLoss
from data import (
    FUSDataset, FUSPairDataset, SyntheticFUSDataset,
    CrossSessionPairDataset, CrossSessionCollator,
    RandomAffine2D, RandomIntensityNoise, RandomCrop,
    RandomElasticDeformation, RandomFlip, RandomGammaCorrection,
    Compose, PairedCompose,
    get_fus_transforms,
)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_augmentation(cfg):
    """
    根据配置构建数据增强。

    对跨 session 数据返回 PairedCompose（几何变换对 moving/fixed 一致，
    强度变换独立）；对其他数据返回普通 Compose。
    """
    aug_cfg = cfg['data'].get('augmentation', {})
    if not aug_cfg.get('enabled', False):
        return None

    data_type = cfg['data'].get('type', 'real')

    geometric_transforms = []
    intensity_transforms = []

    # 仿射变换 → 几何
    if aug_cfg.get('affine', {}).get('enabled', False):
        affine_cfg = aug_cfg['affine']
        geometric_transforms.append(RandomAffine2D(
            rotation=affine_cfg['rotation'],
            translation=affine_cfg['translation'],
            scale=tuple(affine_cfg.get('scale', [0.95, 1.05])),
            p=affine_cfg['p']
        ))

    # 随机裁剪 → 几何
    if aug_cfg.get('crop', {}).get('enabled', False):
        crop_cfg = aug_cfg['crop']
        geometric_transforms.append(RandomCrop(
            crop_scale=tuple(crop_cfg['scale']),
            p=crop_cfg['p']
        ))

    # 弹性变形 → 几何
    if aug_cfg.get('elastic', {}).get('enabled', False):
        elastic_cfg = aug_cfg['elastic']
        geometric_transforms.append(RandomElasticDeformation(
            grid_size=elastic_cfg.get('grid_size', 4),
            magnitude=elastic_cfg.get('magnitude', 8.0),
            p=elastic_cfg.get('p', 0.5)
        ))

    # 水平翻转 → 几何
    if aug_cfg.get('flip', {}).get('enabled', False):
        flip_cfg = aug_cfg['flip']
        geometric_transforms.append(RandomFlip(
            horizontal=True, vertical=False,
            p=flip_cfg.get('p', 0.5)
        ))

    # 强度噪声 → 强度
    if aug_cfg.get('intensity', {}).get('enabled', False):
        int_cfg = aug_cfg['intensity']
        intensity_transforms.append(RandomIntensityNoise(
            noise_std=int_cfg['noise_std'],
            multiplicative_range=tuple(int_cfg['multiplicative_range']),
            brightness_range=tuple(int_cfg['brightness_range']),
            p=int_cfg['p']
        ))

    # Gamma 校正 → 强度
    if aug_cfg.get('gamma', {}).get('enabled', False):
        gamma_cfg = aug_cfg['gamma']
        intensity_transforms.append(RandomGammaCorrection(
            gamma_range=tuple(gamma_cfg.get('range', [0.7, 1.3])),
            p=gamma_cfg.get('p', 0.5)
        ))

    if len(geometric_transforms) == 0 and len(intensity_transforms) == 0:
        return None

    # 跨 session 数据使用 PairedCompose
    if data_type == 'cross_session':
        return PairedCompose(
            geometric_transforms=geometric_transforms,
            intensity_transforms=intensity_transforms,
        )
    else:
        # 其他模式仍用普通 Compose
        return Compose(geometric_transforms + intensity_transforms)


def build_datasets(cfg):
    """构建训练和验证数据集。"""
    data_cfg = cfg['data']
    data_type = data_cfg.get('type', 'real')

    if data_type == 'cross_session':
        # ====== 跨 session 预定义配对模式 ======
        print(f"Using cross-session pairs from: {data_cfg['cross_session_dir']}")

        augmentation = build_augmentation(cfg)

        train_set = CrossSessionPairDataset(
            data_dir=data_cfg['cross_session_dir'],
            split='train',
            mode=data_cfg.get('training_mode', 'B'),
            use_padded=data_cfg.get('use_padded', True),
            normalize=data_cfg.get('normalize', 'percentile'),
            percentile=tuple(data_cfg.get('percentile', [1, 99])),
            apply_log=data_cfg.get('log_transform', True),
            augmentation=augmentation,
            include_reverse=data_cfg.get('include_reverse', True),
        )

        val_set = CrossSessionPairDataset(
            data_dir=data_cfg['cross_session_dir'],
            split='val',
            mode=data_cfg.get('training_mode', 'B'),
            use_padded=data_cfg.get('use_padded', True),
            normalize=data_cfg.get('normalize', 'percentile'),
            percentile=tuple(data_cfg.get('percentile', [1, 99])),
            apply_log=data_cfg.get('log_transform', True),
            augmentation=None,  # 验证集不增强
        )

        print(f"  Train pairs: {len(train_set)}, Val pairs: {len(val_set)}")
        print(f"  Training mode: {data_cfg.get('training_mode', 'B')}")
        print(f"  Normalize: {data_cfg.get('normalize', 'percentile')}, "
              f"Log transform: {data_cfg.get('log_transform', True)}")

        return train_set, val_set, 'cross_session'

    elif data_type == 'synthetic':
        # ====== 合成数据模式 ======
        syn_cfg = data_cfg['synthetic']
        print(f"Using synthetic data: {syn_cfg['n_samples']} samples")

        full_dataset = SyntheticFUSDataset(
            size=syn_cfg['n_samples'],
            image_size=tuple(data_cfg['target_size']),
            motion_type=syn_cfg['motion_type'],
            max_displacement=syn_cfg['max_displacement'],
            noise_level=syn_cfg['noise_level']
        )

        val_size = int(len(full_dataset) * data_cfg['val_split'])
        train_size = len(full_dataset) - val_size
        train_set, val_set = random_split(full_dataset, [train_size, val_size])

        return train_set, val_set, 'synthetic'

    else:
        # ====== 原有真实数据模式 ======
        print(f"Loading real data from: {data_cfg['data_path']}")

        augmentation = build_augmentation(cfg)

        base_dataset = FUSDataset(
            data_path=data_cfg['data_path'],
            target_size=tuple(data_cfg['target_size']),
            normalize=data_cfg.get('normalize', 'percentile'),
            log_transform=data_cfg.get('log_transform', True),
            gaussian_sigma=data_cfg.get('gaussian_sigma'),
            mat_key=data_cfg.get('mat_key', 'data'),
            augmentation=augmentation
        )

        print(f"Loaded {len(base_dataset)} frames")

        pair_dataset = FUSPairDataset(
            base_dataset,
            mode=data_cfg['pair_mode'],
            ref_idx=data_cfg['ref_idx'],
            window_size=data_cfg.get('window_size', 5),
            max_pairs=data_cfg.get('max_pairs')
        )

        print(f"Created {len(pair_dataset)} pairs (mode: {data_cfg['pair_mode']})")

        val_size = int(len(pair_dataset) * data_cfg['val_split'])
        train_size = len(pair_dataset) - val_size
        train_set, val_set = random_split(pair_dataset, [train_size, val_size])

        return train_set, val_set, 'real'


def pretrain_synthetic(model, cfg, device):
    """
    合成数据预训练：在大量合成配对上预训练，学习基本的配准能力。

    使用 NCC + 有监督 flow MSE 双损失。
    """
    pretrain_cfg = cfg.get('pretrain', {})
    if not pretrain_cfg.get('enabled', False):
        return

    n_samples = pretrain_cfg.get('n_samples', 3000)
    epochs = pretrain_cfg.get('epochs', 100)
    lr = pretrain_cfg.get('lr', 1e-4)
    batch_size = pretrain_cfg.get('batch_size', 8)
    flow_weight = pretrain_cfg.get('flow_weight', 0.1)

    print(f"\n{'='*60}")
    print(f"SYNTHETIC PRE-TRAINING: {n_samples} samples, {epochs} epochs")
    print(f"{'='*60}")

    from data import SyntheticFUSDataset

    dataset = SyntheticFUSDataset(
        size=n_samples,
        image_size=(128, 100),
        motion_type='mixed',
        max_displacement=10,
        noise_level=0.05,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )

    sim_loss = NCC(win_size=15)
    reg_loss = Grad(penalty='l2')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for source, target, flow_gt in loader:
            source = source.to(device)
            target = target.to(device)
            flow_gt = flow_gt.to(device)

            outputs = model(source, target)
            warped, flow = outputs[0], outputs[1]

            # NCC 相似度损失
            loss_sim = sim_loss(warped, target)
            # 正则化损失
            loss_reg = reg_loss(flow)
            # 有监督 flow 损失（合成数据有 ground truth）
            loss_flow = torch.nn.functional.mse_loss(flow, flow_gt)

            loss = loss_sim + 0.05 * loss_reg + flow_weight * loss_flow

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Pretrain Epoch {epoch}/{epochs}: loss={avg:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

    print(f"Synthetic pre-training complete.\n")


def build_model(cfg):
    model_cfg = cfg['model']
    return VxmDense2D(
        in_channels=model_cfg['in_channels'],
        enc_channels=model_cfg['enc_channels'],
        dec_channels=model_cfg['dec_channels'],
        integration_steps=model_cfg['integration_steps'],
        bidir=model_cfg.get('bidir', False),
    )


def build_loss(cfg):
    loss_cfg = cfg['loss']
    if loss_cfg['similarity'] == 'ncc':
        sim_loss = NCC(win_size=loss_cfg['ncc_win_size'])
    elif loss_cfg['similarity'] == 'ms_ncc':
        ms_cfg = loss_cfg.get('ms_ncc', {})
        sim_loss = MultiScaleNCC(
            scales=tuple(ms_cfg.get('scales', [1, 2, 4])),
            win_sizes=tuple(ms_cfg.get('win_sizes', [15, 9, 5])),
            weights=tuple(ms_cfg.get('weights', [0.5, 0.3, 0.2])),
        )
    elif loss_cfg['similarity'] == 'mse':
        sim_loss = MSE()
    else:
        raise ValueError(f"Unknown similarity loss: {loss_cfg['similarity']}")

    reg_type = loss_cfg.get('reg_type', 'grad')
    if reg_type == 'diffusion':
        reg_loss = Diffusion()
    elif reg_type == 'bending':
        reg_loss = BendingEnergy()
    else:
        reg_loss = Grad(penalty=loss_cfg.get('reg_penalty', 'l2'))

    return RegistrationLoss(
        sim_loss, reg_loss,
        reg_weight=loss_cfg['reg_weight'],
        bidir_weight=loss_cfg.get('bidir_weight', 0.0),
    )


def train(cfg, resume_path=None):
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 构建数据集
    train_set, val_set, data_mode = build_datasets(cfg)
    print(f"Train pairs: {len(train_set)}, Val pairs: {len(val_set)}")

    # DataLoader 配置
    loader_cfg = cfg.get('dataloader', {})
    use_mask = cfg.get('loss', {}).get('mask_aware', False) and data_mode == 'cross_session'
    is_bidir = cfg.get('model', {}).get('bidir', False)

    collate_fn = CrossSessionCollator() if data_mode == 'cross_session' else None

    train_loader = DataLoader(
        train_set,
        batch_size=loader_cfg.get('batch_size', 4),
        shuffle=loader_cfg.get('shuffle', True),
        num_workers=loader_cfg.get('num_workers', 0),
        pin_memory=loader_cfg.get('pin_memory', False) and torch.cuda.is_available(),
        drop_last=loader_cfg.get('drop_last', False),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=loader_cfg.get('batch_size', 4),
        shuffle=False,
        num_workers=loader_cfg.get('num_workers', 0),
        pin_memory=loader_cfg.get('pin_memory', False) and torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    # 模型
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    if is_bidir:
        print("  Bidirectional training enabled")
    if use_mask:
        print("  Mask-aware loss enabled")

    # 损失和优化器
    criterion = build_loss(cfg)
    train_cfg = cfg['train']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg.get('weight_decay', 0.0)
    )

    # 学习率调度
    scheduler = None
    sched_type = train_cfg.get('lr_scheduler', 'cosine')
    if sched_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg['epochs']
        )
    elif sched_type == 'warmup_cosine':
        warmup_epochs = train_cfg.get('warmup_epochs', 20)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg['epochs'] - warmup_epochs
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    elif sched_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5
        )

    # 从检查点恢复训练
    start_epoch = 1
    best_val_loss = float('inf')
    if resume_path is not None:
        print(f"\nResuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        print(f"  Skipping synthetic pre-training (already done before checkpoint)")
    else:
        # 合成数据预训练（仅首次训练时运行）
        pretrain_synthetic(model, cfg, device)

    # 训练循环
    os.makedirs(train_cfg['save_dir'], exist_ok=True)
    patience_counter = 0
    early_stopping_patience = train_cfg.get('early_stopping')

    # 自适应正则化调度
    reg_schedule = train_cfg.get('reg_schedule', {})
    reg_start = reg_schedule.get('start', criterion.reg_weight)
    reg_end = reg_schedule.get('end', criterion.reg_weight)
    reg_warmup_frac = reg_schedule.get('warmup_fraction', 0.33)

    # 梯度累积
    accumulate_steps = train_cfg.get('accumulate_steps', 1)

    # 日志文件
    log_path = os.path.join(train_cfg['save_dir'], 'training_log.txt')

    # 防止不同训练 run 复用同一 save_dir 导致 checkpoint / log 混杂。
    if resume_path is None:
        existing_artifacts = []
        existing_artifacts.extend(glob.glob(os.path.join(train_cfg['save_dir'], 'checkpoint_epoch*.pth')))
        existing_artifacts.extend(
            path for path in [
                os.path.join(train_cfg['save_dir'], 'best_model.pth'),
                os.path.join(train_cfg['save_dir'], 'best_model_swa.pth'),
                log_path,
            ]
            if os.path.exists(path)
        )
        if existing_artifacts:
            raise RuntimeError(
                "save_dir already contains training artifacts. "
                "Use a new versioned save_dir for a fresh run, or pass --resume "
                "to continue an existing run."
            )

    if resume_path is not None:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\nResuming from checkpoint: {resume_path}\n")

    for epoch in range(start_epoch, train_cfg['epochs'] + 1):
        # 更新正则化权重
        total_epochs = train_cfg['epochs']
        warmup_end = int(total_epochs * reg_warmup_frac)
        if epoch <= warmup_end:
            criterion.reg_weight = reg_start
        else:
            progress = (epoch - warmup_end) / max(total_epochs - warmup_end, 1)
            criterion.reg_weight = reg_start + (reg_end - reg_start) * progress

        model.train()
        epoch_loss = 0.0
        epoch_sim = 0.0
        epoch_reg = 0.0
        t0 = time.time()

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            # 解析 batch（兼容两种数据模式）
            if data_mode == 'cross_session':
                source = batch['moving'].to(device)
                target = batch['fixed'].to(device)
                mask = batch['mask'].to(device) if use_mask else None
            elif isinstance(batch, dict):
                source = batch['moving'].to(device)
                target = batch['fixed'].to(device)
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(device)
            else:
                if len(batch) == 3:
                    source, target, _ = batch
                else:
                    source, target = batch
                source = source.to(device)
                target = target.to(device)
                mask = None

            # Forward pass
            if is_bidir:
                warped, flow, warped_target, neg_flow = model(source, target)
                loss, sim_loss, reg_loss = criterion(
                    warped, target, flow,
                    warped_target=warped_target, source=source,
                    mask=mask,
                )
            else:
                warped, flow = model(source, target)
                loss, sim_loss, reg_loss = criterion(
                    warped, target, flow, mask=mask,
                )

            # 梯度累积：loss 除以累积步数
            (loss / accumulate_steps).backward()

            if (step + 1) % accumulate_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_sim += sim_loss.item()
            epoch_reg += reg_loss.item()

            if (step + 1) % train_cfg['log_interval'] == 0:
                print(f"  Step {step + 1}/{len(train_loader)}: "
                      f"loss={loss.item():.4f} sim={sim_loss.item():.4f} reg={reg_loss.item():.4f}")

        n_steps = max(len(train_loader), 1)
        epoch_loss /= n_steps
        epoch_sim /= n_steps
        epoch_reg /= n_steps

        # 验证
        model.eval()
        val_loss = 0.0
        val_sim = 0.0
        val_reg = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if data_mode == 'cross_session':
                    source = batch['moving'].to(device)
                    target = batch['fixed'].to(device)
                    mask = batch['mask'].to(device) if use_mask else None
                elif isinstance(batch, dict):
                    source = batch['moving'].to(device)
                    target = batch['fixed'].to(device)
                    mask = batch.get('mask', None)
                    if mask is not None:
                        mask = mask.to(device)
                else:
                    if len(batch) == 3:
                        source, target, _ = batch
                    else:
                        source, target = batch
                    source = source.to(device)
                    target = target.to(device)
                    mask = None

                if is_bidir:
                    warped, flow, warped_target, neg_flow = model(source, target)
                    loss, s_loss, r_loss = criterion(
                        warped, target, flow,
                        warped_target=warped_target, source=source,
                        mask=mask,
                    )
                else:
                    warped, flow = model(source, target)
                    loss, s_loss, r_loss = criterion(warped, target, flow, mask=mask)

                val_loss += loss.item()
                val_sim += s_loss.item()
                val_reg += r_loss.item()

        n_val = max(len(val_loader), 1)
        val_loss /= n_val
        val_sim /= n_val
        val_reg /= n_val

        elapsed = time.time() - t0
        lr_str = f" lr={optimizer.param_groups[0]['lr']:.6f}" if scheduler else ""
        log_line = (f"Epoch {epoch}/{train_cfg['epochs']} ({elapsed:.1f}s) - "
                    f"train={epoch_loss:.4f} (sim={epoch_sim:.4f} reg={epoch_reg:.4f}) "
                    f"val={val_loss:.4f} (sim={val_sim:.4f} reg={val_reg:.4f}){lr_str} "
                    f"reg_w={criterion.reg_weight:.4f}")
        print(log_line)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')

        if scheduler:
            scheduler.step()

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(train_cfg['save_dir'], 'best_model.pth'))
            print(f"  -> New best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        # 定期保存检查点
        if epoch % train_cfg['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, os.path.join(train_cfg['save_dir'], f'checkpoint_epoch{epoch}.pth'))

        # Early stopping
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={early_stopping_patience})")
            break

    # SWA（随机权重平均）- 提升泛化能力
    swa_cfg = train_cfg.get('swa', {})
    if swa_cfg.get('enabled', False):
        print(f"\n{'='*60}")
        print("SWA (Stochastic Weight Averaging)")
        print(f"{'='*60}")

        swa_epochs = swa_cfg.get('epochs', 30)
        swa_lr = swa_cfg.get('lr', 1e-5)

        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

        for swa_epoch in range(1, swa_epochs + 1):
            model.train()
            for step, batch in enumerate(train_loader):
                if data_mode == 'cross_session':
                    source = batch['moving'].to(device)
                    target = batch['fixed'].to(device)
                    mask = batch['mask'].to(device) if use_mask else None
                else:
                    if len(batch) == 3:
                        source, target, _ = batch
                    else:
                        source, target = batch
                    source, target = source.to(device), target.to(device)
                    mask = None

                if is_bidir:
                    warped, flow, warped_target, neg_flow = model(source, target)
                    loss, _, _ = criterion(warped, target, flow,
                                           warped_target=warped_target, source=source, mask=mask)
                else:
                    warped, flow = model(source, target)
                    loss, _, _ = criterion(warped, target, flow, mask=mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            swa_model.update_parameters(model)
            swa_scheduler.step()

            if swa_epoch % 5 == 0:
                print(f"  SWA Epoch {swa_epoch}/{swa_epochs}")

        # 更新 BN 统计量
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

        # 保存 SWA 模型
        torch.save(swa_model.module.state_dict(),
                   os.path.join(train_cfg['save_dir'], 'best_model_swa.pth'))
        print("SWA model saved as best_model_swa.pth")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Model saved to: {train_cfg['save_dir']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train fUS-VoxelMorph')
    parser.add_argument('--config', type=str, default='configs/cross_session.yaml',
                        help='Path to config file')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--synthetic', action='store_true',
                        help='Force synthetic data mode')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 应用 CLI 覆盖
    if args.synthetic:
        cfg['data']['type'] = 'synthetic'
    if args.data_path:
        cfg['data']['data_path'] = args.data_path
    if args.epochs:
        cfg['train']['epochs'] = args.epochs
    if args.lr:
        cfg['train']['lr'] = args.lr
    if args.batch_size:
        cfg['dataloader']['batch_size'] = args.batch_size
    if args.device:
        cfg['device'] = args.device

    train(cfg, resume_path=args.resume)
