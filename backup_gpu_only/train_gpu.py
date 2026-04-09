import numpy as np
#!/usr/bin/env python
"""
Training script for fUS-VoxelMorph 2D registration (GPU + Mixed Precision).

Complete training pipeline with:
- Command-line argument parsing
- Data loading (real + synthetic)
- Model initialization
- Training/validation loops with Mixed Precision (AMP)
- TensorBoard logging
- Model checkpointing
- Learning rate scheduling

GPU VERSION: This version uses CUDA with automatic mixed precision (AMP)
for faster training. For CPU training, use train_cpu.py

Usage:
    # Basic training with config file
    python train.py --config configs/default.yaml

    # Override parameters
    python train.py --config configs/default.yaml --epochs 300 --lr 5e-5

    # Use synthetic data
    python train.py --synthetic --batch_size 8

    # Resume training
    python train.py --resume checkpoints/latest_checkpoint.pth
"""

import argparse
import os
import sys
import time
from datetime import datetime

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler

# Import models and losses
from models import VxmDense2D
from losses import NCC, MSE, Grad, Diffusion, RegistrationLoss

# Import data pipeline
from data import (
    FUSDataset, FUSPairDataset, SyntheticFUSDataset,
    get_fus_transforms, Compose
)

# Import utilities
from utils import (
    TensorBoardLogger, ConsoleLogger,
    EarlyStopping, LRScheduler, MetricsTracker, CheckpointManager,
    count_parameters, get_gpu_memory,
    compute_ncc, compute_mse,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train fUS-VoxelMorph for 2D image registration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')

    # Data options
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--data_path', type=str, default=None,
                           help='Path to data directory or file')
    data_group.add_argument('--synthetic', action='store_true',
                           help='Use synthetic data instead of real data')
    data_group.add_argument('--synthetic_samples', type=int, default=1000,
                           help='Number of synthetic samples to generate')
    data_group.add_argument('--pair_mode', type=str, default=None,
                           choices=['consecutive', 'random', 'to_reference', 'sliding_window'],
                           help='Pairing strategy for real data')
    data_group.add_argument('--val_split', type=float, default=0.1,
                           help='Validation split ratio')

    # Model options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--in_channels', type=int, default=1,
                            help='Number of input channels')
    model_group.add_argument('--enc_channels', type=int, nargs='+', default=None,
                            help='Encoder channel dimensions')
    model_group.add_argument('--dec_channels', type=int, nargs='+', default=None,
                            help='Decoder channel dimensions')
    model_group.add_argument('--integration_steps', type=int, default=None,
                            help='Number of integration steps (0 = no diffeomorphic)')

    # Loss options
    loss_group = parser.add_argument_group('Loss Options')
    loss_group.add_argument('--similarity', type=str, default=None,
                           choices=['ncc', 'mse'],
                           help='Similarity loss type')
    loss_group.add_argument('--ncc_win_size', type=int, default=9,
                           help='NCC window size')
    loss_group.add_argument('--reg_type', type=str, default='grad',
                           choices=['grad', 'diffusion'],
                           help='Regularization type')
    loss_group.add_argument('--reg_weight', type=float, default=1.0,
                           help='Regularization weight (lambda)')
    loss_group.add_argument('--reg_penalty', type=str, default='l2',
                           choices=['l1', 'l2'],
                           help='Regularization penalty type')

    # Training options
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--epochs', type=int, default=None,
                            help='Number of training epochs')
    train_group.add_argument('--batch_size', type=int, default=None,
                            help='Batch size')
    train_group.add_argument('--lr', type=float, default=None,
                            help='Learning rate')
    train_group.add_argument('--weight_decay', type=float, default=0.0,
                            help='Weight decay (L2 regularization)')
    train_group.add_argument('--lr_scheduler', type=str, default='cosine',
                            choices=['cosine', 'step', 'plateau', 'exponential', 'warmup_cosine', 'none'],
                            help='Learning rate scheduler')
    train_group.add_argument('--early_stopping', type=int, default=None,
                            help='Early stopping patience (epochs)')

    # Augmentation options
    aug_group = parser.add_argument_group('Data Augmentation')
    aug_group.add_argument('--augment', action='store_true',
                          help='Enable data augmentation')
    aug_group.add_argument('--rotation', type=float, default=5.0,
                          help='Max rotation angle (degrees)')
    aug_group.add_argument('--translation', type=float, default=5.0,
                          help='Max translation (pixels)')
    aug_group.add_argument('--noise_std', type=float, default=0.02,
                          help='Noise standard deviation')

    # Logging and checkpointing
    log_group = parser.add_argument_group('Logging & Checkpointing')
    log_group.add_argument('--save_dir', type=str, default='checkpoints',
                          help='Directory to save checkpoints')
    log_group.add_argument('--log_dir', type=str, default='runs',
                          help='Directory for TensorBoard logs')
    log_group.add_argument('--log_interval', type=int, default=10,
                          help='Log every N batches')
    log_group.add_argument('--vis_interval', type=int, default=50,
                          help='Visualize every N batches')
    log_group.add_argument('--save_interval', type=int, default=20,
                          help='Save checkpoint every N epochs')
    log_group.add_argument('--no_tensorboard', action='store_true',
                          help='Disable TensorBoard logging')
    log_group.add_argument('--num_vis_samples', type=int, default=4,
                          help='Number of samples to visualize')

    # System options
    sys_group = parser.add_argument_group('System Options')
    sys_group.add_argument('--device', type=str, default=None,
                          help='Device to use (cuda/cpu)')
    sys_group.add_argument('--num_workers', type=int, default=4,
                          help='Number of data loading workers')
    sys_group.add_argument('--seed', type=int, default=42,
                          help='Random seed for reproducibility')

    # Resume training
    resume_group = parser.add_argument_group('Resume Training')
    resume_group.add_argument('--resume', type=str, default=None,
                             help='Path to checkpoint to resume from')
    resume_group.add_argument('--start_epoch', type=int, default=1,
                             help='Starting epoch (for resuming)')

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config(args, cfg):
    """Merge command-line arguments with config file."""
    # Command-line args override config file
    def get_value(key, default=None):
        arg_val = getattr(args, key, None)
        if arg_val is not None:
            return arg_val
        return cfg.get(key, default)

    # Build merged config
    merged = {
        # Data
        'data_path': get_value('data_path', 'data/fus_frames/'),
        'synthetic': args.synthetic,
        'synthetic_samples': get_value('synthetic_samples', 1000),
        'pair_mode': get_value('pair_mode', 'consecutive'),
        'val_split': get_value('val_split', 0.1),

        # Model
        'in_channels': get_value('in_channels', 1),
        'enc_channels': get_value('enc_channels', [16, 32, 32, 32]),
        'dec_channels': get_value('dec_channels', [32, 32, 32, 32, 16, 16]),
        'integration_steps': get_value('integration_steps', 7),

        # Loss
        'similarity': get_value('similarity', 'ncc'),
        'ncc_win_size': get_value('ncc_win_size', 9),
        'reg_type': get_value('reg_type', 'grad'),
        'reg_weight': get_value('reg_weight', 1.0),
        'reg_penalty': get_value('reg_penalty', 'l2'),

        # Training
        'epochs': get_value('epochs', 200),
        'batch_size': get_value('batch_size', 8),
        'lr': get_value('lr', 1e-4),
        'weight_decay': get_value('weight_decay', 0.0),
        'lr_scheduler': get_value('lr_scheduler', 'cosine'),
        'early_stopping': args.early_stopping,

        # Augmentation
        'augment': args.augment,
        'rotation': get_value('rotation', 5.0),
        'translation': get_value('translation', 5.0),
        'noise_std': get_value('noise_std', 0.02),

        # Logging
        'save_dir': get_value('save_dir', 'checkpoints'),
        'log_dir': get_value('log_dir', 'runs'),
        'log_interval': get_value('log_interval', 10),
        'vis_interval': get_value('vis_interval', 50),
        'save_interval': get_value('save_interval', 20),
        'use_tensorboard': not args.no_tensorboard,
        'num_vis_samples': get_value('num_vis_samples', 4),

        # System
        'device': get_value('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        'num_workers': get_value('num_workers', 4),
        'seed': get_value('seed', 42),
    }

    return merged


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_augmentation(cfg):
    """Build data augmentation pipeline."""
    if not cfg['augment']:
        return None

    transforms = []

    if cfg['rotation'] > 0 or cfg['translation'] > 0:
        from data import RandomAffine2D
        transforms.append(RandomAffine2D(
            rotation=cfg['rotation'],
            translation=cfg['translation'],
            p=0.5
        ))

    if cfg['noise_std'] > 0:
        from data import RandomIntensityNoise
        transforms.append(RandomIntensityNoise(
            noise_std=cfg['noise_std'],
            p=0.5
        ))

    if len(transforms) == 0:
        return None

    return Compose(transforms)


def build_datasets(cfg, logger):
    """Build training and validation datasets."""
    logger.info("Building datasets...")

    if cfg['synthetic']:
        logger.info(f"Using synthetic data: {cfg['synthetic_samples']} samples")
        full_dataset = SyntheticFUSDataset(
            size=cfg['synthetic_samples'],
            image_size=(128, 100),
            motion_type='mixed',
            max_displacement=10,
            noise_level=0.05
        )

        # Split train/val
        val_size = int(len(full_dataset) * cfg['val_split'])
        train_size = len(full_dataset) - val_size
        train_set, val_set = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg['seed'])
        )

    else:
        # Real data
        if not os.path.exists(cfg['data_path']):
            raise FileNotFoundError(f"Data path not found: {cfg['data_path']}")

        logger.info(f"Loading real data from: {cfg['data_path']}")

        # Build augmentation
        augmentation = build_augmentation(cfg)

        # Base dataset
        base_dataset = FUSDataset(
            data_path=cfg['data_path'],
            target_size=(128, 100),
            normalize='minmax',
            augmentation=augmentation
        )

        logger.info(f"  Loaded {len(base_dataset)} frames")

        # Pair dataset
        pair_dataset = FUSPairDataset(
            base_dataset,
            mode=cfg['pair_mode'],
            ref_idx=0
        )

        logger.info(f"  Created {len(pair_dataset)} pairs (mode: {cfg['pair_mode']})")

        # Split
        val_size = int(len(pair_dataset) * cfg['val_split'])
        train_size = len(pair_dataset) - val_size
        train_set, val_set = random_split(
            pair_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg['seed'])
        )

    logger.info(f"  Train: {len(train_set)}, Val: {len(val_set)}")
    return train_set, val_set


def build_model(cfg):
    """Build the registration model."""
    model = VxmDense2D(
        in_channels=cfg['in_channels'],
        enc_channels=cfg['enc_channels'],
        dec_channels=cfg['dec_channels'],
        integration_steps=cfg['integration_steps'],
    )
    return model


def build_loss(cfg):
    """Build the training loss."""
    # Similarity loss
    if cfg['similarity'] == 'ncc':
        sim_loss = NCC(win_size=cfg['ncc_win_size'])
    elif cfg['similarity'] == 'mse':
        sim_loss = MSE()
    else:
        raise ValueError(f"Unknown similarity loss: {cfg['similarity']}")

    # Regularization loss
    if cfg['reg_type'] == 'diffusion':
        reg_loss = Diffusion()
    else:
        reg_loss = Grad(penalty=cfg['reg_penalty'])

    # Combined loss
    criterion = RegistrationLoss(
        sim_loss, reg_loss,
        reg_weight=cfg['reg_weight']
    )

    return criterion


def train_epoch(model, dataloader, criterion, optimizer, scaler, device,
                cfg, logger, tb_logger, epoch):
    """Train for one epoch with mixed precision."""
    model.train()

    metrics = {
        'loss': 0.0,
        'sim_loss': 0.0,
        'reg_loss': 0.0,
    }

    num_batches = len(dataloader)
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Parse batch
        if len(batch) == 3:  # (source, target, flow_gt)
            source, target, _ = batch
        else:  # (source, target)
            source, target = batch

        source = source.to(device)
        target = target.to(device)

        # Forward pass with mixed precision
        with autocast(device_type=device.type):
            warped, flow = model(source, target)
            loss, sim_loss, reg_loss = criterion(warped, target, flow)

        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        metrics['loss'] += loss.item()
        metrics['sim_loss'] += sim_loss.item()
        metrics['reg_loss'] += reg_loss.item()

        # Logging
        if (batch_idx + 1) % cfg['log_interval'] == 0:
            avg_loss = metrics['loss'] / (batch_idx + 1)
            logger.info(
                f"  Batch [{batch_idx+1}/{num_batches}] "
                f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) "
                f"Sim: {sim_loss.item():.4f} Reg: {reg_loss.item():.4f}"
            )

        # Visualization
        if tb_logger.enabled and (batch_idx + 1) % cfg['vis_interval'] == 0:
            tb_logger.log_registration_visualization(
                'train/registration',
                source[:cfg['num_vis_samples']],
                target[:cfg['num_vis_samples']],
                warped[:cfg['num_vis_samples']],
                flow[:cfg['num_vis_samples']],
                step=epoch * num_batches + batch_idx,
                max_samples=cfg['num_vis_samples']
            )

    # Average metrics
    for key in metrics:
        metrics[key] /= num_batches

    elapsed = time.time() - start_time
    metrics['time'] = elapsed

    return metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device, cfg, tb_logger, epoch):
    """Validate the model."""
    model.eval()

    metrics = {
        'loss': 0.0,
        'sim_loss': 0.0,
        'reg_loss': 0.0,
        'ncc': 0.0,
        'mse': 0.0,
    }

    all_sources = []
    all_targets = []
    all_warped = []
    all_flows = []

    num_batches = len(dataloader)

    for batch in dataloader:
        if len(batch) == 3:
            source, target, _ = batch
        else:
            source, target = batch

        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Forward pass with mixed precision
        with autocast(device_type=device.type):
            warped, flow = model(source, target)
            loss, sim_loss, reg_loss = criterion(warped, target, flow)

        # Accumulate metrics
        metrics['loss'] += loss.item()
        metrics['sim_loss'] += sim_loss.item()
        metrics['reg_loss'] += reg_loss.item()

        # Additional metrics
        for i in range(source.shape[0]):
            src_np = source[i, 0].cpu().numpy()
            tgt_np = target[i, 0].cpu().numpy()
            warp_np = warped[i, 0].cpu().numpy()

            metrics['ncc'] += compute_ncc(warp_np, tgt_np)
            metrics['mse'] += compute_mse(warp_np, tgt_np)

        # Store for visualization
        all_sources.append(source)
        all_targets.append(target)
        all_warped.append(warped)
        all_flows.append(flow)

    # Average metrics
    for key in metrics:
        metrics[key] /= (num_batches * source.shape[0] if key in ['ncc', 'mse']
                        else num_batches)

    # Visualize validation results
    if tb_logger.enabled:
        # Concatenate first batch
        sources = torch.cat(all_sources[:1], dim=0)[:cfg['num_vis_samples']]
        targets = torch.cat(all_targets[:1], dim=0)[:cfg['num_vis_samples']]
        warped = torch.cat(all_warped[:1], dim=0)[:cfg['num_vis_samples']]
        flows = torch.cat(all_flows[:1], dim=0)[:cfg['num_vis_samples']]

        tb_logger.log_registration_visualization(
            'val/registration',
            sources, targets, warped, flows,
            step=epoch,
            max_samples=cfg['num_vis_samples']
        )

    return metrics


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load and merge config
    cfg_file = load_config(args.config)
    cfg = merge_config(args, cfg_file)

    # Set seed
    set_seed(cfg['seed'])

    # Create directories
    os.makedirs(cfg['save_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)

    # Create loggers
    console_logger = ConsoleLogger(
        log_file=os.path.join(cfg['save_dir'], 'training.log')
    )
    console_logger.section("fUS-VoxelMorph Training")

    # Print configuration
    console_logger.info("Configuration:")
    for key, value in sorted(cfg.items()):
        console_logger.info(f"  {key}: {value}")

    # Setup device - GPU VERSION
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console_logger.info(f"\n[GPU VERSION] Using device: {device}")
    if device.type == 'cuda':
        console_logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        console_logger.info(f"CUDA Version: {torch.version.cuda}")
        # Enable cuDNN auto-tuner for better performance
        torch.backends.cudnn.benchmark = True
        console_logger.info("Enabled cuDNN benchmark for optimal performance")
    else:
        console_logger.warning("WARNING: CUDA not available, falling back to CPU")

    # TensorBoard logger
    experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(cfg['log_dir'], experiment_name)
    tb_logger = TensorBoardLogger(
        log_dir=log_dir,
        enabled=cfg['use_tensorboard']
    )
    if tb_logger.enabled:
        console_logger.info(f"TensorBoard logs: {log_dir}")
        console_logger.info(f"View with: tensorboard --logdir={cfg['log_dir']}")

    # Build datasets
    train_set, val_set = build_datasets(cfg, console_logger)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,  # GPU version always uses pin_memory
        persistent_workers=True if cfg['num_workers'] > 0 else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True,  # GPU version always uses pin_memory
        persistent_workers=True if cfg['num_workers'] > 0 else False
    )

    # Build model
    console_logger.info("\nBuilding model...")
    model = build_model(cfg).to(device)
    param_info = count_parameters(model)
    console_logger.info(f"  Parameters: {param_info['trainable']:,} ({param_info['trainable_mb']:.2f} MB)")

    # Build loss (will be on GPU via input tensors)
    criterion = build_loss(cfg)
    console_logger.info(f"  Loss: {cfg['similarity']} + {cfg['reg_weight']} * {cfg['reg_type']}")

    # Build optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )
    console_logger.info(f"  Optimizer: Adam (lr={cfg['lr']}, wd={cfg['weight_decay']})")

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(device=device.type)
    console_logger.info("  Mixed Precision: Enabled (torch.amp)")

    # Build LR scheduler
    scheduler = LRScheduler(
        optimizer,
        mode=cfg['lr_scheduler'],
        T_max=cfg['epochs']
    )
    console_logger.info(f"  LR Scheduler: {cfg['lr_scheduler']}")

    # Build checkpoint manager
    ckpt_manager = CheckpointManager(
        save_dir=cfg['save_dir'],
        keep_last_n=3
    )

    # Build early stopping
    early_stopping = None
    if cfg['early_stopping'] is not None:
        early_stopping = EarlyStopping(patience=cfg['early_stopping'])
        console_logger.info(f"  Early Stopping: patience={cfg['early_stopping']}")

    # Resume training if specified
    start_epoch = 1
    best_val_loss = float('inf')

    if args.resume:
        console_logger.info(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = ckpt_manager.load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['metrics'].get('val_loss', float('inf'))
        console_logger.info(f"  Resuming from epoch {start_epoch}")

    # Training loop
    console_logger.section("Starting Training (GPU + Mixed Precision)")
    
    # Log initial GPU memory
    if device.type == 'cuda':
        initial_mem = get_gpu_memory()
        console_logger.info(f"Initial GPU Memory: {initial_mem['allocated']:.1f} MB allocated")

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        console_logger.info(f"\nEpoch {epoch}/{cfg['epochs']}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            cfg, console_logger, tb_logger, epoch
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, cfg, tb_logger, epoch
        )

        # Update learning rate
        scheduler.step(epoch, val_metrics['loss'])
        current_lr = scheduler.get_lr()

        # Print summary with GPU memory
        gpu_mem_str = ""
        if device.type == 'cuda':
            gpu_mem = get_gpu_memory()
            gpu_mem_str = f" GPU: {gpu_mem['allocated']:.0f}MB"
        console_logger.info(
            f"  Train - Loss: {train_metrics['loss']:.4f} "
            f"(Sim: {train_metrics['sim_loss']:.4f}, Reg: {train_metrics['reg_loss']:.4f}) "
            f"Time: {train_metrics['time']:.1f}s{gpu_mem_str}"
        )
        console_logger.info(
            f"  Val   - Loss: {val_metrics['loss']:.4f} "
            f"(Sim: {val_metrics['sim_loss']:.4f}, Reg: {val_metrics['reg_loss']:.4f}) "
            f"NCC: {val_metrics['ncc']:.4f} MSE: {val_metrics['mse']:.4f} "
            f"LR: {current_lr:.6f}"
        )

        # Log to TensorBoard
        if tb_logger.enabled:
            tb_logger.log_scalars('train', {
                'loss': train_metrics['loss'],
                'sim_loss': train_metrics['sim_loss'],
                'reg_loss': train_metrics['reg_loss'],
                'time': train_metrics['time'],
            }, epoch)

            tb_logger.log_scalars('val', {
                'loss': val_metrics['loss'],
                'sim_loss': val_metrics['sim_loss'],
                'reg_loss': val_metrics['reg_loss'],
                'ncc': val_metrics['ncc'],
                'mse': val_metrics['mse'],
            }, epoch)

            tb_logger.log_scalar('lr', current_lr, epoch)

            # Log GPU memory
            gpu_mem = get_gpu_memory()
            if gpu_mem:
                tb_logger.log_scalar('gpu/memory_allocated_mb', gpu_mem['allocated'], epoch)
                tb_logger.log_scalar('gpu/memory_reserved_mb', gpu_mem['reserved'], epoch)
                tb_logger.log_scalar('gpu/max_allocated_mb', gpu_mem['max_allocated'], epoch)

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            console_logger.info(f"  New best model! (val_loss={val_metrics['loss']:.4f})")

        # Regular checkpoint
        if epoch % cfg['save_interval'] == 0 or is_best:
            ckpt_manager.save_checkpoint(
                epoch, model, optimizer, scheduler,
                {'train_loss': train_metrics['loss'], 'val_loss': val_metrics['loss']},
                is_best=is_best
            )

        # Save latest for resume
        ckpt_manager.save_latest(
            epoch, model, optimizer, scheduler,
            {'train_loss': train_metrics['loss'], 'val_loss': val_metrics['loss']}
        )

        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_metrics['loss']):
                console_logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                break

    # Training complete
    console_logger.section("Training Complete")
    console_logger.info(f"Best validation loss: {best_val_loss:.4f}")
    console_logger.info(f"Checkpoints saved to: {cfg['save_dir']}")
    if tb_logger.enabled:
        console_logger.info(f"TensorBoard logs: {log_dir}")

    tb_logger.close()


if __name__ == '__main__':
    main()
