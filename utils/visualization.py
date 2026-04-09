"""
Visualization utilities for fUS registration results.

Includes:
- Registration result overview
- Green/magenta overlay comparison (like Zhong et al. Fig.3)
- Displacement field visualization (quiver + magnitude)
- Jacobian determinant analysis
- Difference maps
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch


def create_registration_figure(source, target, warped, flow=None, metrics=None,
                                title=None, figsize=(20, 5)):
    """
    Create a comprehensive registration overview figure.

    Parameters
    ----------
    source : np.ndarray (H, W)
        Moving image
    target : np.ndarray (H, W)
        Fixed/target image
    warped : np.ndarray (H, W)
        Warped/moving image
    flow : np.ndarray (2, H, W), optional
        Displacement field
    metrics : dict, optional
        Dictionary of metrics to display
    title : str, optional
        Figure title
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    has_flow = flow is not None
    n_cols = 5 if has_flow else 4

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    # Source (Moving)
    axes[0].imshow(source, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Source (Moving)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Target (Fixed)
    axes[1].imshow(target, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Target (Fixed)', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Warped
    axes[2].imshow(warped, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Warped', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Overlay comparison
    overlay = create_overlay_image(source, target, warped, mode='green_magenta')
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (G:Before, M:After)', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    # Flow magnitude
    if has_flow:
        flow_mag = np.sqrt(flow[0]**2 + flow[1]**2)
        im = axes[4].imshow(flow_mag, cmap='viridis')
        axes[4].set_title(f'|Flow| (max={flow_mag.max():.1f}px)', fontsize=12, fontweight='bold')
        axes[4].axis('off')
        plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    # Add metrics text
    if metrics:
        metric_text = format_metrics_text(metrics)
        fig.text(0.5, 0.02, metric_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    if metrics:
        plt.subplots_adjust(bottom=0.15)

    return fig


def create_overlay_image(source, target, warped, mode='green_magenta'):
    """
    Create overlay image showing before/after registration.

    Modes:
    - 'green_magenta': Green=before, Magenta=after (like Zhong et al.)
    - 'red_green': Red=before, Green=after
    - 'checkerboard': Checkerboard pattern

    Parameters
    ----------
    source, target, warped : np.ndarray (H, W)
    mode : str
        Overlay mode

    Returns
    -------
    overlay : np.ndarray (H, W, 3)
        RGB overlay image
    """
    H, W = source.shape

    if mode == 'green_magenta':
        # Green = before registration (source vs target)
        # Magenta = after registration (warped vs target)
        # Yellow = good match (target)

        overlay = np.zeros((H, W, 3))

        # Normalize images
        s = np.clip(source, 0, 1)
        t = np.clip(target, 0, 1)
        w = np.clip(warped, 0, 1)

        # Create RGB channels
        # R: target
        # G: source (before) and warped (after)
        # B: target

        # Before: source vs target -> green where source, magenta where both
        diff_before = np.abs(s - t)
        diff_after = np.abs(w - t)

        # Better match after -> more magenta (R+B)
        # Worse match after -> more green (G)

        # Use difference to determine color
        # Small diff_after = good registration = magenta (R=1, G=0, B=1 weighted by target)
        # Large diff_after = poor registration = green (G=1 weighted by source)

        r = t * (1 - diff_after)  # Magenta component
        g = diff_before * (1 - diff_after) + s * diff_after  # Green for before mismatch
        b = t * (1 - diff_after)

        overlay[:, :, 0] = np.clip(r + diff_after * 0.5, 0, 1)
        overlay[:, :, 1] = np.clip(g + t * 0.3, 0, 1)
        overlay[:, :, 2] = np.clip(b + diff_after * 0.5, 0, 1)

    elif mode == 'red_green':
        # Red = before, Green = after
        overlay = np.zeros((H, W, 3))
        overlay[:, :, 0] = source  # Red channel = before
        overlay[:, :, 1] = warped  # Green channel = after
        overlay[:, :, 2] = target * 0.5  # Blue channel = reference

    elif mode == 'checkerboard':
        # Checkerboard pattern alternating before/after
        checker = np.zeros((H, W, 3))
        block_size = 16

        for i in range(0, H, block_size):
            for j in range(0, W, block_size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    checker[i:i+block_size, j:j+block_size, :] = \
                        np.stack([source[i:i+block_size, j:j+block_size]] * 3, axis=-1)
                else:
                    checker[i:i+block_size, j:j+block_size, :] = \
                        np.stack([warped[i:i+block_size, j:j+block_size]] * 3, axis=-1)

        overlay = checker

    return np.clip(overlay, 0, 1)


def create_overlay_comparison(source, target, warped, save_path=None, figsize=(15, 5)):
    """
    Create detailed overlay comparison figure.

    Similar to Zhong et al. Fig.3 style visualization.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Before registration overlay (source + target)
    before_overlay = np.zeros((*source.shape, 3))
    before_overlay[:, :, 0] = target  # Red = target
    before_overlay[:, :, 1] = source  # Green = source
    before_overlay[:, :, 2] = target * 0.3

    axes[0].imshow(np.clip(before_overlay, 0, 1))
    axes[0].set_title('Before Registration\n(R: Fixed, G: Moving)', fontsize=11)
    axes[0].axis('off')

    # After registration overlay (warped + target)
    after_overlay = np.zeros((*source.shape, 3))
    after_overlay[:, :, 0] = target  # Red = target
    after_overlay[:, :, 1] = warped  # Green = warped
    after_overlay[:, :, 2] = target * 0.3

    axes[1].imshow(np.clip(after_overlay, 0, 1))
    axes[1].set_title('After Registration\n(R: Fixed, G: Warped)', fontsize=11)
    axes[1].axis('off')

    # Green/Magenta overlay (Zhong et al. style)
    # Yellow = good match, Green/Magenta = mismatch
    diff_before = np.abs(source - target)
    diff_after = np.abs(warped - target)

    gm_overlay = np.zeros((*source.shape, 3))
    gm_overlay[:, :, 0] = target * (1 - diff_after) + diff_after * 0.8  # R
    gm_overlay[:, :, 1] = target + diff_before * 0.5  # G
    gm_overlay[:, :, 2] = target * (1 - diff_after) + diff_after * 0.3  # B

    axes[2].imshow(np.clip(gm_overlay, 0, 1))
    axes[2].set_title('Improvement Overlay\n(G: Before, M: After)', fontsize=11)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def create_flow_visualization(flow, source=None, save_path=None, step=8,
                               figsize=(14, 6)):
    """
    Create comprehensive displacement field visualization.

    Parameters
    ----------
    flow : np.ndarray (2, H, W)
        Displacement field [dx, dy]
    source : np.ndarray (H, W), optional
        Source image for background
    save_path : str, optional
    step : int
        Subsampling step for quiver plot
    figsize : tuple
    """
    H, W = flow.shape[1], flow.shape[2]
    flow_mag = np.sqrt(flow[0]**2 + flow[1]**2)
    flow_dir = np.arctan2(flow[1], flow[0])

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Magnitude heatmap
    if source is not None:
        axes[0].imshow(source, cmap='gray', alpha=0.3)

    im1 = axes[0].imshow(flow_mag, cmap='viridis', alpha=0.8 if source is not None else 1.0)
    axes[0].set_title(f'Displacement Magnitude\n(mean={flow_mag.mean():.2f}px, max={flow_mag.max():.2f}px)')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Direction colormap
    im2 = axes[1].imshow(flow_dir, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title('Displacement Direction')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, ticks=[-np.pi, 0, np.pi],
                format=plt.FuncFormatter(lambda x, pos: ['-π', '0', 'π'][pos]))

    # Quiver plot
    y, x = np.mgrid[0:H:step, 0:W:step]
    u = flow[0, ::step, ::step]
    v = flow[1, ::step, ::step]

    if source is not None:
        axes[2].imshow(source, cmap='gray', alpha=0.5)

    # Color by magnitude
    mag_sampled = np.sqrt(u**2 + v**2)
    quiver = axes[2].quiver(x, y, u, -v, mag_sampled, cmap='plasma',
                            scale=50, width=0.003, headwidth=3)
    axes[2].set_xlim(0, W)
    axes[2].set_ylim(H, 0)
    axes[2].set_aspect('equal')
    axes[2].set_title('Displacement Vectors')
    axes[2].axis('off')
    plt.colorbar(quiver, ax=axes[2], fraction=0.046, label='Magnitude (px)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def create_jacobian_visualization(flow, save_path=None, threshold=0,
                                  figsize=(14, 5)):
    """
    Create Jacobian determinant analysis visualization.

    Parameters
    ----------
    flow : np.ndarray (2, H, W)
        Displacement field
    save_path : str, optional
    threshold : float
        Threshold for highlighting problematic regions
    figsize : tuple
    """
    from utils.metrics import jacobian_determinant_2d

    jac_det, stats = jacobian_determinant_2d(flow)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Jacobian determinant heatmap
    im1 = axes[0].imshow(jac_det, cmap='RdYlGn', vmin=0, vmax=2,
                         interpolation='nearest')
    axes[0].set_title(f'Jacobian Determinant det(J)\n'
                      f'mean={stats["mean"]:.3f}, std={stats["std"]:.3f}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Negative Jacobian regions (folding)
    neg_mask = jac_det <= threshold
    neg_overlay = np.zeros((*jac_det.shape, 4))  # RGBA
    neg_overlay[neg_mask] = [1, 0, 0, 0.5]  # Red with transparency

    axes[1].imshow(jac_det, cmap='gray', alpha=0.5)
    axes[1].imshow(neg_overlay)
    axes[1].set_title(f'Folding Regions (|det|≤{threshold})\n'
                      f'{stats["pct_neg"]:.2f}% of voxels')
    axes[1].axis('off')

    # Jacobian histogram
    axes[2].hist(jac_det.flatten(), bins=50, color='steelblue', edgecolor='black')
    axes[2].axvline(x=0, color='r', linestyle='--', linewidth=2, label='det(J)=0')
    axes[2].axvline(x=stats['mean'], color='g', linestyle='--', linewidth=2,
                   label=f'mean={stats["mean"]:.3f}')
    axes[2].set_xlabel('Jacobian Determinant')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of det(J)')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def create_difference_map(source, target, warped, save_path=None, figsize=(15, 5)):
    """
    Create difference maps before and after registration.

    Parameters
    ----------
    source, target, warped : np.ndarray (H, W)
    save_path : str, optional
    figsize : tuple
    """
    diff_before = np.abs(source - target)
    diff_after = np.abs(warped - target)
    improvement = diff_before - diff_after

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Before
    im1 = axes[0].imshow(diff_before, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title(f'|Moving - Fixed|\nmean={diff_before.mean():.4f}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # After
    im2 = axes[1].imshow(diff_after, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f'|Warped - Fixed|\nmean={diff_after.mean():.4f}')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # Improvement
    im3 = axes[2].imshow(improvement, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
    axes[2].set_title(f'Improvement\n(G: better, R: worse)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    # Histogram of improvement
    axes[3].hist(improvement.flatten(), bins=50, color='steelblue', edgecolor='black')
    axes[3].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[3].axvline(x=improvement.mean(), color='g', linestyle='--', linewidth=2,
                   label=f'mean={improvement.mean():.4f}')
    axes[3].set_xlabel('Pixel-wise Improvement')
    axes[3].set_ylabel('Frequency')
    axes[3].set_title('Improvement Distribution')
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def create_grid_overlay(flow, source=None, grid_spacing=16, save_path=None,
                        figsize=(12, 6)):
    """
    Create grid overlay showing deformation.

    Parameters
    ----------
    flow : np.ndarray (2, H, W)
    source : np.ndarray (H, W), optional
    grid_spacing : int
    save_path : str, optional
    """
    H, W = flow.shape[1], flow.shape[2]

    # Create regular grid
    y_grid, x_grid = np.mgrid[0:H:grid_spacing, 0:W:grid_spacing]

    # Warp grid points
    y_warped = y_grid + flow[1, y_grid, x_grid]
    x_warped = x_grid + flow[0, y_grid, x_grid]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Original grid
    if source is not None:
        axes[0].imshow(source, cmap='gray', alpha=0.5)

    for i in range(y_grid.shape[0]):
        axes[0].plot(x_grid[i, :], y_grid[i, :], 'b-', alpha=0.5, linewidth=0.5)
    for j in range(y_grid.shape[1]):
        axes[0].plot(x_grid[:, j], y_grid[:, j], 'b-', alpha=0.5, linewidth=0.5)

    axes[0].set_xlim(0, W)
    axes[0].set_ylim(H, 0)
    axes[0].set_aspect('equal')
    axes[0].set_title('Regular Grid')
    axes[0].axis('off')

    # Warped grid
    if source is not None:
        axes[1].imshow(source, cmap='gray', alpha=0.3)

    for i in range(y_warped.shape[0]):
        axes[1].plot(x_warped[i, :], y_warped[i, :], 'r-', alpha=0.7, linewidth=0.5)
    for j in range(y_warped.shape[1]):
        axes[1].plot(x_warped[:, j], y_warped[:, j], 'r-', alpha=0.7, linewidth=0.5)

    axes[1].set_xlim(0, W)
    axes[1].set_ylim(H, 0)
    axes[1].set_aspect('equal')
    axes[1].set_title('Deformed Grid')
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def save_all_visualizations(source, target, warped, flow, output_dir,
                            prefix='', format='png', dpi=150):
    """
    Generate and save all visualization types.

    Parameters
    ----------
    source, target, warped : np.ndarray (H, W)
    flow : np.ndarray (2, H, W)
    output_dir : str
    prefix : str
    format : str
    dpi : int
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Registration overview
    create_registration_figure(
        source, target, warped, flow,
        title='Registration Result'
    )
    plt.savefig(os.path.join(output_dir, f'{prefix}overview.{format}'),
                dpi=dpi, bbox_inches='tight')
    plt.close()

    # 2. Overlay comparison
    create_overlay_comparison(
        source, target, warped,
        save_path=os.path.join(output_dir, f'{prefix}overlay.{format}')
    )

    # 3. Flow visualization
    create_flow_visualization(
        flow, source=source,
        save_path=os.path.join(output_dir, f'{prefix}flow.{format}')
    )

    # 4. Jacobian analysis
    create_jacobian_visualization(
        flow,
        save_path=os.path.join(output_dir, f'{prefix}jacobian.{format}')
    )

    # 5. Difference map
    create_difference_map(
        source, target, warped,
        save_path=os.path.join(output_dir, f'{prefix}difference.{format}')
    )

    # 6. Grid overlay
    create_grid_overlay(
        flow, source=source,
        save_path=os.path.join(output_dir, f'{prefix}grid.{format}')
    )


def format_metrics_text(metrics):
    """Format metrics dictionary as text."""
    lines = []

    if 'ncc_before' in metrics and 'ncc_after' in metrics:
        lines.append(f"NCC: {metrics['ncc_before']:.4f} → {metrics['ncc_after']:.4f} "
                    f"(Δ {metrics['ncc_improvement']:+.4f})")

    if 'mse_before' in metrics and 'mse_after' in metrics:
        lines.append(f"MSE: {metrics['mse_before']:.4f} → {metrics['mse_after']:.4f} "
                    f"(Δ {metrics['mse_improvement']:+.4f})")

    if 'flow_mean' in metrics:
        lines.append(f"Flow: mean={metrics['flow_mean']:.2f}px, "
                    f"max={metrics['flow_max']:.2f}px")

    if 'jac_pct_neg' in metrics:
        lines.append(f"Jacobian: {metrics['jac_pct_neg']:.2f}% folding")

    return '  |  '.join(lines)


# Legacy function for backward compatibility
def plot_registration_result(source, target, warped, flow=None, save_path=None):
    """Simple registration result plot (legacy API)."""
    fig = create_registration_figure(source, target, warped, flow)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_displacement_field(flow, save_path, step=4):
    """Save displacement field as quiver plot (legacy API)."""
    create_flow_visualization(flow, save_path=save_path, step=step)
