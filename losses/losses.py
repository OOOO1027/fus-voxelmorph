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


class MultiScaleNCC(nn.Module):
    """
    Multi-scale Normalized Cross-Correlation loss.

    在多个分辨率级别计算 NCC，提供从粗到细的相似度信号：
    - 低分辨率：捕捉全局结构对齐（大感受野）
    - 高分辨率：捕捉精细血管对齐（局部精度）

    Parameters
    ----------
    scales : list of int
        下采样倍率，例如 [1, 2, 4] 表示原始、1/2、1/4 分辨率
    win_sizes : list of int
        每个尺度对应的 NCC 窗口大小
    weights : list of float
        每个尺度的权重
    eps : float
        数值稳定性常量
    """

    def __init__(self, scales=(1, 2, 4), win_sizes=(15, 9, 5),
                 weights=(0.5, 0.3, 0.2), eps=1e-5):
        super().__init__()
        assert len(scales) == len(win_sizes) == len(weights)
        self.scales = scales
        self.ncc_modules = nn.ModuleList([NCC(win_size=w, eps=eps) for w in win_sizes])
        self.weights = weights

    def forward(self, y_pred, y_true, mask=None):
        total_loss = 0.0
        for scale, ncc, weight in zip(self.scales, self.ncc_modules, self.weights):
            if scale == 1:
                pred_s = y_pred
                true_s = y_true
                mask_s = mask
            else:
                pred_s = F.avg_pool2d(y_pred, kernel_size=scale, stride=scale)
                true_s = F.avg_pool2d(y_true, kernel_size=scale, stride=scale)
                mask_s = F.avg_pool2d(mask, kernel_size=scale, stride=scale) if mask is not None else None
                # 二值化 mask（pooling 后可能不是 0/1）
                if mask_s is not None:
                    mask_s = (mask_s > 0.5).float()
            total_loss = total_loss + weight * ncc(pred_s, true_s, mask=mask_s)
        return total_loss


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
                loss_dy = (dy.pow(2) * mask_dy).sum() / mask_dy.sum().clamp(min=1.0)
                loss_dx = (dx.pow(2) * mask_dx).sum() / mask_dx.sum().clamp(min=1.0)
            else:
                loss_dy = (dy.abs() * mask_dy).sum() / mask_dy.sum().clamp(min=1.0)
                loss_dx = (dx.abs() * mask_dx).sum() / mask_dx.sum().clamp(min=1.0)
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
            loss = ((d2y.pow(2) * m_d2y).sum() / m_d2y.sum().clamp(min=1.0)
                    + (d2x.pow(2) * m_d2x).sum() / m_d2x.sum().clamp(min=1.0)
                    + 2 * (dxy.pow(2) * m_dxy).sum() / m_dxy.sum().clamp(min=1.0))
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
            return ((d2y.pow(2) * m_d2y).sum() / m_d2y.sum().clamp(min=1.0)
                    + (d2x.pow(2) * m_d2x).sum() / m_d2x.sum().clamp(min=1.0)
                    + 2 * (dxy.pow(2) * m_dxy).sum() / m_dxy.sum().clamp(min=1.0))

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
