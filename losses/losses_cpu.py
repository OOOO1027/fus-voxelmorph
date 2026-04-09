"""
Loss functions for 2D fUS image registration.

Training loss = Similarity(warped, target) + lambda * Regularization(flow)

Similarity losses compare image content after warping.
Regularization losses penalise non-smooth displacement fields.

References
----------
[1] VoxelMorph — Balakrishnan et al., IEEE TMI 2019.
[2] An Unsupervised Learning Model for Deformable Brain Registration —
    Dalca et al., MICCAI 2018.
[3] fUS motion correction — Zhong et al., local NCC + Jacobian analysis.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_channel_mean(values, mask):
    """Average masked per-pixel values across both spatial sites and channels."""
    num_channels = values.shape[1]
    normalizer = mask.sum().clamp(min=1.0) * num_channels
    return (values * mask).sum() / normalizer


# ═══════════════════════════════════════════════════════════════════════════
#  Similarity losses
# ═══════════════════════════════════════════════════════════════════════════

class NCC(nn.Module):
    """
    Local (windowed) Normalized Cross-Correlation loss.

    For each spatial position, NCC is computed inside a square window of
    side ``win_size``.  The loss returned is the negative mean of the local
    NCC map, so *minimising* the loss *maximises* similarity.

    This is the standard formulation used in VoxelMorph and in fUS motion
    correction literature (Zhong et al.).  A small window (default 9×9)
    suits fUS power Doppler images where vascular structures are fine.

    Parameters
    ----------
    win_size : int
        Side length of the square local window.  Default 9.
        Typical choices: 7 for high-res, 9 for standard fUS FOV,
        11–15 for smoother similarity landscapes.
    eps : float
        Small constant to avoid division by zero in flat regions.

    Shape
    -----
    Input : (B, C, H, W) — any number of channels; each channel is
            treated independently.
    mask  : (B, 1, H, W) optional — valid region mask (1=valid, 0=padding).
            When provided, only valid-region pixels contribute to the loss.
    Output: scalar (mean over all positions, channels and batch).
    """

    def __init__(self, win_size=9, eps=1e-5):
        super().__init__()
        self.win_size = win_size
        self.eps = eps

    def forward(self, y_pred, y_true, mask=None):
        B, C, H, W = y_true.shape
        win = self.win_size
        pad = win // 2
        n = win * win

        # Uniform summing kernel — shape (C, 1, win, win) for groups=C
        weight = torch.ones(C, 1, win, win,
                            device=y_true.device, dtype=y_true.dtype)

        # Local sums via grouped convolution (each channel independent)
        sum_I  = F.conv2d(y_true,         weight, padding=pad, groups=C)
        sum_J  = F.conv2d(y_pred,         weight, padding=pad, groups=C)
        sum_I2 = F.conv2d(y_true * y_true, weight, padding=pad, groups=C)
        sum_J2 = F.conv2d(y_pred * y_pred, weight, padding=pad, groups=C)
        sum_IJ = F.conv2d(y_true * y_pred, weight, padding=pad, groups=C)

        # Local means
        mu_I = sum_I / n
        mu_J = sum_J / n

        # Local (co)variances — using E[X²] - E[X]² form
        var_I  = sum_I2 / n - mu_I * mu_I
        var_J  = sum_J2 / n - mu_J * mu_J
        cov_IJ = sum_IJ / n - mu_I * mu_J

        # NCC² per position = cov² / (var_I * var_J)
        # Clamp variances to zero (can go slightly negative from numerics)
        cc = cov_IJ * cov_IJ / (var_I.clamp(min=0) * var_J.clamp(min=0)
                                 + self.eps)

        if mask is not None:
            # mask: (B, 1, H, W) -> broadcast to (B, C, H, W)
            cc = cc * mask
            # 仅在有效区域内求均值
            num_valid = mask.sum().clamp(min=1.0) * C
            return -(cc.sum() / num_valid)

        return -cc.mean()


class MSE(nn.Module):
    """Mean Squared Error similarity loss, with optional mask support."""

    def forward(self, y_pred, y_true, mask=None):
        if mask is not None:
            diff_sq = (y_pred - y_true) ** 2 * mask
            return diff_sq.sum() / mask.sum().clamp(min=1.0)
        return F.mse_loss(y_pred, y_true)


# ═══════════════════════════════════════════════════════════════════════════
#  Regularization losses
# ═══════════════════════════════════════════════════════════════════════════

class Grad(nn.Module):
    """
    Spatial-gradient regularization on the displacement field.

    Penalises the first-order finite differences of the displacement field
    to encourage spatially smooth deformations and prevent folding.

    This is the standard "diffusion-like" regulariser used in VoxelMorph
    (Eq. 2 in Balakrishnan et al. 2019).

    Parameters
    ----------
    penalty : str
        ``'l2'`` — sum of squared differences (default, smooth penalty).
        ``'l1'`` — sum of absolute differences (edge-preserving).

    Shape
    -----
    Input : (B, 2, H, W)  displacement field.
    Output: scalar.
    """

    def __init__(self, penalty='l2'):
        super().__init__()
        assert penalty in ('l1', 'l2')
        self.penalty = penalty

    def forward(self, flow, mask=None):
        # First-order finite differences along each spatial axis
        dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]   # (B, 2, H-1, W)
        dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]    # (B, 2, H, W-1)

        if mask is not None:
            # 对 mask 做相应裁剪，只在有效区域内计算正则
            mask_dy = (mask[:, :, 1:, :] * mask[:, :, :-1, :])  # (B, 1, H-1, W)
            mask_dx = (mask[:, :, :, 1:] * mask[:, :, :, :-1])  # (B, 1, H, W-1)

            if self.penalty == 'l2':
                loss_dy = _masked_channel_mean(dy.pow(2), mask_dy)
                loss_dx = _masked_channel_mean(dx.pow(2), mask_dx)
            else:
                loss_dy = _masked_channel_mean(dy.abs(), mask_dy)
                loss_dx = _masked_channel_mean(dx.abs(), mask_dx)
            return (loss_dy + loss_dx) / 2.0

        if self.penalty == 'l2':
            loss = (dy.pow(2).mean() + dx.pow(2).mean()) / 2.0
        else:
            loss = (dy.abs().mean() + dx.abs().mean()) / 2.0

        return loss


class Diffusion(nn.Module):
    """
    Diffusion regularization (Laplacian penalty).

    Penalises the *second-order* spatial derivatives of the displacement
    field, encouraging globally smoother deformations than first-order
    ``Grad``.  Particularly useful in the diffeomorphic (VecInt) setting
    where the velocity field should be very smooth.

    Implements:  L = mean( |d²φ/dx²|² + 2|d²φ/dxdy|² + |d²φ/dy²|² )

    Shape
    -----
    Input : (B, 2, H, W)  displacement (or velocity) field.
    Output: scalar.
    """

    def forward(self, flow, mask=None):
        # Second-order finite differences
        # d²/dy²
        d2y = flow[:, :, 2:, :] - 2 * flow[:, :, 1:-1, :] + flow[:, :, :-2, :]
        # d²/dx²
        d2x = flow[:, :, :, 2:] - 2 * flow[:, :, :, 1:-1] + flow[:, :, :, :-2]
        # d²/dxdy  (mixed partial)
        dxy = (flow[:, :, 1:, 1:] - flow[:, :, 1:, :-1]
               - flow[:, :, :-1, 1:] + flow[:, :, :-1, :-1])

        if mask is not None:
            m_d2y = mask[:, :, 2:, :] * mask[:, :, 1:-1, :] * mask[:, :, :-2, :]
            m_d2x = mask[:, :, :, 2:] * mask[:, :, :, 1:-1] * mask[:, :, :, :-2]
            m_dxy = mask[:, :, 1:, 1:] * mask[:, :, 1:, :-1] * mask[:, :, :-1, 1:] * mask[:, :, :-1, :-1]
            loss = (_masked_channel_mean(d2y.pow(2), m_d2y)
                    + _masked_channel_mean(d2x.pow(2), m_d2x)
                    + 2 * _masked_channel_mean(dxy.pow(2), m_dxy))
            return loss / 4.0

        loss = d2y.pow(2).mean() + d2x.pow(2).mean() + 2 * dxy.pow(2).mean()
        return loss / 4.0


class BendingEnergy(nn.Module):
    """
    Bending energy regularization.

    Higher-order smoothness penalty that is the continuous analogue of thin
    plate spline energy. Even smoother than Diffusion — useful when the
    deformation must be very regular (e.g. rigid-body dominant fUS motion).

    L = mean( (d²φ/dy²)² + (d²φ/dx²)² + 2*(d²φ/dxdy)² )

    Note: numerically identical to ``Diffusion`` for 2D but conceptually
    distinct (bending energy has a different weighting in 3D); included for
    completeness and clarity.

    Shape
    -----
    Input : (B, 2, H, W).
    Output: scalar.
    """

    def forward(self, flow, mask=None):
        d2y = flow[:, :, 2:, :] - 2 * flow[:, :, 1:-1, :] + flow[:, :, :-2, :]
        d2x = flow[:, :, :, 2:] - 2 * flow[:, :, :, 1:-1] + flow[:, :, :, :-2]
        dxy = (flow[:, :, 1:, 1:] - flow[:, :, 1:, :-1]
               - flow[:, :, :-1, 1:] + flow[:, :, :-1, :-1])

        if mask is not None:
            m_d2y = mask[:, :, 2:, :] * mask[:, :, 1:-1, :] * mask[:, :, :-2, :]
            m_d2x = mask[:, :, :, 2:] * mask[:, :, :, 1:-1] * mask[:, :, :, :-2]
            m_dxy = mask[:, :, 1:, 1:] * mask[:, :, 1:, :-1] * mask[:, :, :-1, 1:] * mask[:, :, :-1, :-1]
            return (_masked_channel_mean(d2y.pow(2), m_d2y)
                    + _masked_channel_mean(d2x.pow(2), m_d2x)
                    + 2 * _masked_channel_mean(dxy.pow(2), m_dxy))

        return d2y.pow(2).mean() + d2x.pow(2).mean() + 2 * dxy.pow(2).mean()


# ═══════════════════════════════════════════════════════════════════════════
#  Combined loss
# ═══════════════════════════════════════════════════════════════════════════

class RegistrationLoss(nn.Module):
    """
    Total training loss = sim_loss(warped, target) + lambda * reg_loss(flow).

    Optionally supports bidirectional training with a second similarity
    term on the inverse-warped target.

    Parameters
    ----------
    sim_loss : nn.Module
        Similarity criterion (NCC, MSE …).
    reg_loss : nn.Module
        Regularization criterion (Grad, Diffusion …).
    reg_weight : float
        λ — regularization weight.  Higher → smoother fields.
    bidir_weight : float
        Weight for the inverse (target→source) similarity term.
        0 disables bidirectional loss.

    Returns
    -------
    total : scalar
    sim   : scalar  (detached, for logging)
    reg   : scalar  (detached, for logging)
    """

    def __init__(self, sim_loss, reg_loss, reg_weight=1.0, bidir_weight=0.0):
        super().__init__()
        self.sim_loss = sim_loss
        self.reg_loss = reg_loss
        self.reg_weight = reg_weight
        self.bidir_weight = bidir_weight

    def forward(self, warped, target, flow,
                warped_target=None, source=None, mask=None):
        sim = self.sim_loss(warped, target, mask=mask)

        if self.bidir_weight > 0 and warped_target is not None and source is not None:
            sim_inv = self.sim_loss(warped_target, source, mask=mask)
            sim = sim + self.bidir_weight * sim_inv

        reg = self.reg_loss(flow, mask=mask)
        total = sim + self.reg_weight * reg

        return total, sim.detach(), reg.detach()
