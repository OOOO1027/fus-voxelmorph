"""
Training utilities for fUS-VoxelMorph.

Includes:
- Learning rate schedulers
- Early stopping
- Model checkpointing
- Metric tracking
"""

import os
import json
from collections import defaultdict

import torch
import numpy as np


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.

    Parameters
    ----------
    patience : int
        How many epochs to wait before stopping when loss is not improving
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement
    mode : str
        'min' for loss (lower is better), 'max' for accuracy (higher is better)
    """

    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.is_better = lambda score, best: score < best - min_delta
        else:
            self.is_better = lambda score, best: score > best + min_delta

    def __call__(self, score):
        """
        Check if should stop.

        Returns
        -------
        bool : True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class LRScheduler:
    """
    Learning rate scheduler wrapper with multiple strategies.

    Supports: 'cosine', 'step', 'plateau', 'exponential', 'none'
    """

    def __init__(self, optimizer, mode='cosine', **kwargs):
        """
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
        mode : str
            Scheduler type: 'cosine', 'step', 'plateau', 'exponential', 'none'
        **kwargs :
            Additional arguments for specific schedulers
        """
        self.optimizer = optimizer
        self.mode = mode
        self.scheduler = None

        if mode == 'cosine':
            # Cosine annealing
            T_max = kwargs.get('T_max', 100)
            eta_min = kwargs.get('eta_min', 1e-6)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )

        elif mode == 'step':
            # Step decay
            step_size = kwargs.get('step_size', 50)
            gamma = kwargs.get('gamma', 0.5)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )

        elif mode == 'plateau':
            # Reduce on plateau
            factor = kwargs.get('factor', 0.5)
            patience = kwargs.get('patience', 10)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience,
                verbose=True
            )

        elif mode == 'exponential':
            # Exponential decay
            gamma = kwargs.get('gamma', 0.95)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma
            )

        elif mode == 'warmup_cosine':
            # Warmup + cosine annealing
            warmup_epochs = kwargs.get('warmup_epochs', 10)
            T_max = kwargs.get('T_max', 100)
            self.warmup_epochs = warmup_epochs
            self.base_lr = optimizer.defaults['lr']
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max - warmup_epochs, eta_min=1e-6
            )
            self.current_epoch = 0

        elif mode == 'none':
            self.scheduler = None

    def step(self, epoch=None, metric=None):
        """Step the scheduler."""
        if self.scheduler is None:
            return

        if self.mode == 'plateau':
            # Plateau scheduler needs validation metric
            if metric is not None:
                self.scheduler.step(metric)
        elif self.mode == 'warmup_cosine':
            self.current_epoch += 1
            if self.current_epoch <= self.warmup_epochs:
                # Linear warmup
                lr = self.base_lr * self.current_epoch / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                self.scheduler.step()
        else:
            self.scheduler.step()

    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class MetricsTracker:
    """Track and aggregate metrics during training."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = {}

    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)

    def average(self):
        """Compute average of all tracked metrics."""
        return {key: np.mean(values) for key, values in self.metrics.items()}

    def std(self):
        """Compute standard deviation of all tracked metrics."""
        return {key: np.std(values) for key, values in self.metrics.items()}

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()

    def summary(self):
        """Get a summary string of metrics."""
        avg = self.average()
        return ', '.join([f"{k}={v:.4f}" for k, v in avg.items()])

    def save(self, path):
        """Save metrics history to JSON."""
        with open(path, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)


class CheckpointManager:
    """
    Manage model checkpoints.

    Saves:
    - Best model based on validation metric
    - Regular checkpoints every N epochs
    - Latest checkpoint for resuming
    """

    def __init__(self, save_dir, keep_last_n=3):
        """
        Parameters
        ----------
        save_dir : str
            Directory to save checkpoints
        keep_last_n : int
            Number of recent checkpoints to keep ( deletes older ones)
        """
        self.save_dir = save_dir
        self.keep_last_n = keep_last_n
        self.checkpoint_history = []

        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self, epoch, model, optimizer, scheduler, metrics,
                        is_best=False, filename=None):
        """
        Save a checkpoint.

        Parameters
        ----------
        epoch : int
        model : nn.Module
        optimizer : Optimizer
        scheduler : LRScheduler or None
        metrics : dict
            Current metrics
        is_best : bool
            Whether this is the best model so far
        filename : str, optional
            Custom filename, default is 'checkpoint_epoch{epoch}.pth'
        """
        if filename is None:
            filename = f'checkpoint_epoch{epoch}.pth'

        filepath = os.path.join(self.save_dir, filename)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }

        if scheduler is not None:
            if hasattr(scheduler, 'scheduler') and scheduler.scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.scheduler.state_dict()

        torch.save(checkpoint, filepath)

        # Track checkpoint
        self.checkpoint_history.append(filepath)

        # Clean up old checkpoints
        if len(self.checkpoint_history) > self.keep_last_n:
            old_checkpoint = self.checkpoint_history.pop(0)
            if os.path.exists(old_checkpoint) and 'best' not in old_checkpoint:
                os.remove(old_checkpoint)

        # Save best model separately
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

        return filepath

    def save_latest(self, epoch, model, optimizer, scheduler, metrics):
        """Save as latest checkpoint (overwrites previous)."""
        return self.save_checkpoint(
            epoch, model, optimizer, scheduler, metrics,
            filename='latest_checkpoint.pth'
        )

    def load_checkpoint(self, filepath, model, optimizer=None, scheduler=None):
        """
        Load a checkpoint.

        Returns
        -------
        dict : checkpoint data
        """
        checkpoint = torch.load(filepath, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            if hasattr(scheduler, 'scheduler') and scheduler.scheduler is not None:
                scheduler.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint

    def load_best(self, model, optimizer=None, scheduler=None):
        """Load the best checkpoint."""
        best_path = os.path.join(self.save_dir, 'best_model.pth')
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"No best model found at {best_path}")
        return self.load_checkpoint(best_path, model, optimizer, scheduler)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name=''):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        'trainable': trainable,
        'total': total,
        'trainable_mb': trainable * 4 / (1024 ** 2),  # Assuming float32
        'total_mb': total * 4 / (1024 ** 2)
    }


def get_gpu_memory():
    """Get GPU memory usage."""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / (1024 ** 2),  # MB
            'reserved': torch.cuda.memory_reserved() / (1024 ** 2),
            'max_allocated': torch.cuda.max_memory_allocated() / (1024 ** 2)
        }
    return None
