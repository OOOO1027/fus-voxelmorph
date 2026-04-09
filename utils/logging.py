"""
Logging utilities for training visualization.

Supports TensorBoard for:
- Loss curves (train/val)
- Learning rate scheduling
- Registration visualization (moving, fixed, warped, flow)
- Metrics over time
"""

import os
import io
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

import torch


class TensorBoardLogger:
    """
    TensorBoard logger for training visualization.

    Logs:
    - Scalars: losses, metrics, learning rate
    - Images: registration results (moving, fixed, warped, flow)
    - Histograms: model weights, gradients (optional)
    """

    def __init__(self, log_dir, enabled=True):
        """
        Parameters
        ----------
        log_dir : str
            Directory to save logs
        enabled : bool
            Whether to enable logging (set False to disable without code changes)
        """
        self.enabled = enabled
        self.log_dir = log_dir
        self.writer = None

        if not enabled:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.has_tensorboard = True
        except ImportError:
            print("Warning: tensorboard not installed. Logging disabled.")
            print("Install with: pip install tensorboard")
            self.has_tensorboard = False
            self.enabled = False

    def log_scalar(self, tag, value, step):
        """Log a scalar value."""
        if not self.enabled or not self.has_tensorboard:
            return
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars at once."""
        if not self.enabled or not self.has_tensorboard:
            return
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(self, tag, image, step):
        """
        Log a single image.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Image array (H, W) or (H, W, C) or (C, H, W)
        """
        if not self.enabled or not self.has_tensorboard:
            return

        # Convert to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # Normalize to [0, 255] if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Handle different shapes
        if image.ndim == 2:
            image = image[np.newaxis, :, :]  # (1, H, W)
        elif image.ndim == 3 and image.shape[-1] in [1, 3, 4]:
            image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        self.writer.add_image(tag, image, step)

    def log_images(self, tag, images, step):
        """
        Log a batch of images as a grid.

        Parameters
        ----------
        images : list of np.ndarray or torch.Tensor
            List of images
        """
        if not self.enabled or not self.has_tensorboard:
            return

        # Convert all to numpy
        img_list = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if img.ndim == 3 and img.shape[0] == 1:
                img = img[0]  # (1, H, W) -> (H, W)
            img_list.append(img)

        self.writer.add_images(tag, np.stack(img_list), step, dataformats='NCHW')

    def log_registration_visualization(self, tag, moving, fixed, warped,
                                        flow=None, step=0, max_samples=4):
        """
        Log a comprehensive registration visualization.

        Creates a figure with:
        - Moving image
        - Fixed image
        - Warped image
        - Difference before (|moving - fixed|)
        - Difference after (|warped - fixed|)
        - Displacement magnitude (if flow provided)

        Parameters
        ----------
        moving, fixed, warped : torch.Tensor or np.ndarray
            Images of shape (B, C, H, W) or (B, H, W) or (H, W)
        flow : torch.Tensor or np.ndarray, optional
            Displacement field of shape (B, 2, H, W) or (2, H, W)
        step : int
            Global step
        max_samples : int
            Maximum number of samples to visualize from batch
        """
        if not self.enabled or not self.has_tensorboard:
            return

        # Convert to numpy and extract first samples
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            # Ensure 4D: (B, C, H, W) or (B, 1, H, W)
            if x.ndim == 2:
                x = x[np.newaxis, np.newaxis, :, :]
            elif x.ndim == 3:
                if x.shape[0] == 2:  # Flow
                    x = x[np.newaxis, :, :, :]
                else:
                    x = x[:, np.newaxis, :, :]
            return x

        moving = to_numpy(moving)
        fixed = to_numpy(fixed)
        warped = to_numpy(warped)

        batch_size = min(moving.shape[0], max_samples)

        for i in range(batch_size):
            m = moving[i, 0]  # (H, W)
            f = fixed[i, 0]
            w = warped[i, 0]

            # Create figure
            ncols = 6 if flow is not None else 5
            fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3))

            # Moving
            axes[0].imshow(m, cmap='hot')
            axes[0].set_title('Moving')
            axes[0].axis('off')

            # Fixed
            axes[1].imshow(f, cmap='hot')
            axes[1].set_title('Fixed')
            axes[1].axis('off')

            # Warped
            axes[2].imshow(w, cmap='hot')
            axes[2].set_title('Warped')
            axes[2].axis('off')

            # Difference before
            diff_before = np.abs(m - f)
            axes[3].imshow(diff_before, cmap='gray', vmin=0, vmax=1)
            axes[3].set_title(f'Diff Before\n(mean={diff_before.mean():.3f})')
            axes[3].axis('off')

            # Difference after
            diff_after = np.abs(w - f)
            axes[4].imshow(diff_after, cmap='gray', vmin=0, vmax=1)
            axes[4].set_title(f'Diff After\n(mean={diff_after.mean():.3f})')
            axes[4].axis('off')

            # Flow magnitude
            if flow is not None:
                flow_np = to_numpy(flow)[i]  # (2, H, W)
                flow_mag = np.sqrt(flow_np[0] ** 2 + flow_np[1] ** 2)
                im = axes[5].imshow(flow_mag, cmap='viridis')
                axes[5].set_title(f'|Flow|\n(max={flow_mag.max():.1f}px)')
                axes[5].axis('off')
                plt.colorbar(im, ax=axes[5], fraction=0.046)

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            img_array = np.array(img)

            # Log
            self.log_image(f"{tag}/sample_{i}", img_array, step)

            plt.close(fig)
            buf.close()

    def log_histogram(self, tag, values, step):
        """Log a histogram of values."""
        if not self.enabled or not self.has_tensorboard:
            return
        self.writer.add_histogram(tag, values, step)

    def log_model_graph(self, model, input_sample):
        """Log model graph."""
        if not self.enabled or not self.has_tensorboard:
            return
        try:
            self.writer.add_graph(model, input_sample)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")

    def close(self):
        """Close the logger."""
        if self.writer is not None:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ConsoleLogger:
    """Simple console logger with formatting."""

    def __init__(self, log_file=None):
        self.log_file = log_file
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            # Clear existing log
            open(log_file, 'w').close()

    def log(self, message, level='INFO'):
        """Log a message to console and optionally to file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"[{timestamp}] [{level}] {message}"

        print(formatted)

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted + '\n')

    def info(self, message):
        self.log(message, 'INFO')

    def warning(self, message):
        self.log(message, 'WARN')

    def error(self, message):
        self.log(message, 'ERROR')

    def section(self, title):
        """Print a section header."""
        self.log("=" * 60)
        self.log(title)
        self.log("=" * 60)
