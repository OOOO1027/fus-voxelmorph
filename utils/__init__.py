"""
Utility modules for fUS-VoxelMorph.
"""

# Metrics and visualization
from .metrics import (
    compute_ncc, compute_mse, compute_ssim, compute_ms_ssim,
    compute_dsc, jacobian_determinant_2d, compute_haar_psi,
    compute_all_metrics,
)

from .visualization import (
    plot_registration_result,
    save_displacement_field,
    create_registration_figure,
    create_overlay_image,
    create_overlay_comparison,
    create_flow_visualization,
    create_jacobian_visualization,
    create_difference_map,
    create_grid_overlay,
    save_all_visualizations,
    format_metrics_text,
)

# Training utilities
from .logging import TensorBoardLogger, ConsoleLogger
from .training import (
    EarlyStopping, LRScheduler, MetricsTracker, CheckpointManager,
    AverageMeter, count_parameters, get_gpu_memory,
)

__all__ = [
    # Metrics
    'compute_ncc', 'compute_mse', 'compute_ssim', 'compute_ms_ssim',
    'compute_dsc', 'jacobian_determinant_2d', 'compute_haar_psi',
    'compute_all_metrics',
    # Visualization
    'plot_registration_result', 'save_displacement_field',
    'create_registration_figure', 'create_overlay_image',
    'create_overlay_comparison', 'create_flow_visualization',
    'create_jacobian_visualization', 'create_difference_map',
    'create_grid_overlay', 'save_all_visualizations',
    'format_metrics_text',
    # Logging
    'TensorBoardLogger', 'ConsoleLogger',
    # Training
    'EarlyStopping', 'LRScheduler', 'MetricsTracker', 'CheckpointManager',
    'AverageMeter', 'count_parameters', 'get_gpu_memory',
]
