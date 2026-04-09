"""
Evaluation metrics for 2D fUS image registration.

These are *not* used during training — they are computed in evaluation
scripts to quantify registration quality.

Metrics
-------
- NCC     : Normalized Cross-Correlation (global)
- MSE     : Mean Squared Error
- SSIM    : Structural Similarity Index (local-window)
- MS-SSIM : Multi-Scale Structural Similarity
- DSC     : Dice Similarity Coefficient (on binary masks)
- Jacobian: Determinant statistics (mean, std, % folding)
- HaarPSI : Haar wavelet-based Perceptual Similarity Index

All functions accept plain NumPy arrays (H, W) unless noted otherwise.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# ═══════════════════════════════════════════════════════════════════════════
#  Basic metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_ncc(img1, img2):
    """
    Global Normalized Cross-Correlation.

    Returns
    -------
    float in [-1, 1].  1 = identical (up to affine intensity).
    """
    a = (img1 - img1.mean()).astype(np.float64).ravel()
    b = (img2 - img2.mean()).astype(np.float64).ravel()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-10:
        return 0.0
    return float((a * b).sum() / denom)


def compute_mse(img1, img2):
    """Mean Squared Error."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


# ═══════════════════════════════════════════════════════════════════════════
#  SSIM / MS-SSIM
# ═══════════════════════════════════════════════════════════════════════════

def _gaussian_kernel_1d(size, sigma):
    """1-D Gaussian kernel (normalised)."""
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-0.5 * (coords / sigma) ** 2)
    return g / g.sum()


def _ssim_components(img1, img2, win_size=11, sigma=1.5, data_range=1.0):
    """
    Compute per-pixel SSIM luminance * contrast * structure components
    and return the maps plus means needed for MS-SSIM.

    Uses separable Gaussian weighting (matches Wang et al. 2003).
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Gaussian-weighted local statistics (separable for speed)
    k = _gaussian_kernel_1d(win_size, sigma)

    def filt(x):
        # Separable 2-D Gaussian filtering
        out = np.apply_along_axis(lambda row: np.convolve(row, k, mode='valid'), 1, x)
        out = np.apply_along_axis(lambda col: np.convolve(col, k, mode='valid'), 0, out)
        return out

    mu1 = filt(img1)
    mu2 = filt(img2)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    sigma1_sq = filt(img1 * img1) - mu1_sq
    sigma2_sq = filt(img2 * img2) - mu2_sq
    sigma12   = filt(img1 * img2) - mu12

    # Clamp negative variance from numerics
    sigma1_sq = np.maximum(sigma1_sq, 0.0)
    sigma2_sq = np.maximum(sigma2_sq, 0.0)

    # luminance, contrast-structure
    luminance = (2 * mu12 + C1) / (mu1_sq + mu2_sq + C1)
    cs        = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    ssim_map = luminance * cs

    return ssim_map, luminance, cs


def compute_ssim(img1, img2, win_size=11, sigma=1.5, data_range=1.0):
    """
    Structural Similarity Index (windowed Gaussian-weighted).

    Matches the original Wang et al. 2003 formulation.

    Parameters
    ----------
    img1, img2 : (H, W) arrays.
    win_size : int, Gaussian window support (default 11).
    sigma : float, Gaussian std (default 1.5).
    data_range : float, dynamic range of the data (default 1.0).

    Returns
    -------
    float in [-1, 1].  1 = identical.
    """
    ssim_map, _, _ = _ssim_components(img1, img2, win_size, sigma, data_range)
    return float(ssim_map.mean())


def compute_ms_ssim(img1, img2, weights=None, win_size=11, sigma=1.5,
                    data_range=1.0):
    """
    Multi-Scale SSIM (Wang et al. 2003).

    Downsamples by factor 2 at each scale and combines the contrast-
    structure (CS) term from coarser scales with the luminance + CS from
    the finest scale.

    Parameters
    ----------
    img1, img2 : (H, W) arrays, values in [0, data_range].
    weights : array-like of length n_scales.
        Per-scale exponents.  Default 5-scale weights from the paper:
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333].
    win_size : int
    sigma : float
    data_range : float

    Returns
    -------
    float in [0, 1].  1 = identical.
    """
    if weights is None:
        weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    n_levels = len(weights)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    cs_per_level = []
    for level in range(n_levels):
        ssim_map, lum, cs = _ssim_components(img1, img2, win_size, sigma,
                                             data_range)
        cs_per_level.append(cs.mean())

        if level < n_levels - 1:
            # Average-pool 2× downsample
            h, w = img1.shape
            img1 = (img1[0:h - h % 2:2, 0:w - w % 2:2] +
                    img1[1:h - h % 2:2, 0:w - w % 2:2] +
                    img1[0:h - h % 2:2, 1:w - w % 2:2] +
                    img1[1:h - h % 2:2, 1:w - w % 2:2]) / 4.0
            img2 = (img2[0:h - h % 2:2, 0:w - w % 2:2] +
                    img2[1:h - h % 2:2, 0:w - w % 2:2] +
                    img2[0:h - h % 2:2, 1:w - w % 2:2] +
                    img2[1:h - h % 2:2, 1:w - w % 2:2]) / 4.0

    # The last level uses the full SSIM (luminance × CS)
    # Replace the last CS with full SSIM
    cs_per_level[-1] = ssim_map.mean()

    # Clamp to avoid negative bases with fractional exponents
    cs_per_level = np.array(cs_per_level)
    cs_per_level = np.maximum(cs_per_level, 1e-10)

    ms_ssim = np.prod(cs_per_level ** weights)
    return float(ms_ssim)


# ═══════════════════════════════════════════════════════════════════════════
#  Dice Similarity Coefficient
# ═══════════════════════════════════════════════════════════════════════════

def compute_dsc(img1, img2, threshold=0.5):
    """
    Dice Similarity Coefficient on binarised images.

    Useful for evaluating whether registration preserves vessel masks.
    Images are binarised at ``threshold`` before computing DSC.

    Parameters
    ----------
    img1, img2 : (H, W) arrays, values in [0, 1].
    threshold : float.

    Returns
    -------
    float in [0, 1].  1 = perfect overlap.
    """
    mask1 = (img1 > threshold).astype(bool)
    mask2 = (img2 > threshold).astype(bool)

    intersection = np.sum(mask1 & mask2)
    total = np.sum(mask1) + np.sum(mask2)

    if total == 0:
        return 1.0  # both empty → perfect agreement
    return float(2.0 * intersection / total)


# ═══════════════════════════════════════════════════════════════════════════
#  Jacobian determinant analysis
# ═══════════════════════════════════════════════════════════════════════════

def jacobian_determinant_2d(displacement):
    """
    Compute the Jacobian determinant of a 2D displacement field.

    The transformation is  T(p) = p + displacement(p).
    The Jacobian of T is  J = I + ∇(displacement).
    det(J) < 0 indicates spatial folding (non-diffeomorphic).

    Parameters
    ----------
    displacement : (2, H, W) or (H, W, 2) array.

    Returns
    -------
    jac_det : (H-1, W-1) array of Jacobian determinant values.
    stats : dict with keys:
        'mean'        — mean det(J)
        'std'         — std  det(J)
        'pct_neg'     — % of positions with det(J) ≤ 0 (folding)
        'min'         — minimum det(J)
        'num_neg'     — absolute count of folded voxels
    """
    if displacement.shape[-1] == 2:
        displacement = displacement.transpose(2, 0, 1)

    D = displacement.astype(np.float64)

    # Finite-difference Jacobian of the transformation T = id + D
    # ∂T_x/∂x = 1 + ∂D_x/∂x     ∂T_x/∂y = ∂D_x/∂y
    # ∂T_y/∂x = ∂D_y/∂x          ∂T_y/∂y = 1 + ∂D_y/∂y
    dTx_dx = 1.0 + (D[0, :, 1:] - D[0, :, :-1])   # (H, W-1)
    dTx_dy = D[0, 1:, :] - D[0, :-1, :]             # (H-1, W)
    dTy_dx = D[1, :, 1:] - D[1, :, :-1]             # (H, W-1)
    dTy_dy = 1.0 + (D[1, 1:, :] - D[1, :-1, :])    # (H-1, W)

    # det(J) on the interior (H-1, W-1) grid
    jac_det = (dTx_dx[:-1, :] * dTy_dy[:, :-1]
               - dTx_dy[:, :-1] * dTy_dx[:-1, :])

    neg_mask = jac_det <= 0
    stats = {
        'mean':    float(jac_det.mean()),
        'std':     float(jac_det.std()),
        'min':     float(jac_det.min()),
        'pct_neg': float(neg_mask.sum() / jac_det.size * 100),
        'num_neg': int(neg_mask.sum()),
    }

    return jac_det, stats


# ═══════════════════════════════════════════════════════════════════════════
#  HaarPSI — Haar wavelet Perceptual Similarity Index
# ═══════════════════════════════════════════════════════════════════════════

def _haar_wavelet_2d(img):
    """
    One level of the 2D Haar wavelet transform.

    Returns (LL, LH, HL, HH) sub-bands, each (H/2, W/2).
    """
    h, w = img.shape
    h = h - h % 2
    w = w - w % 2
    img = img[:h, :w]

    # Row-wise: low (average) and high (difference)
    row_lo = (img[:, 0::2] + img[:, 1::2]) / 2.0
    row_hi = (img[:, 0::2] - img[:, 1::2]) / 2.0

    # Column-wise on each
    LL = (row_lo[0::2, :] + row_lo[1::2, :]) / 2.0
    LH = (row_lo[0::2, :] - row_lo[1::2, :]) / 2.0
    HL = (row_hi[0::2, :] + row_hi[1::2, :]) / 2.0
    HH = (row_hi[0::2, :] - row_hi[1::2, :]) / 2.0

    return LL, LH, HL, HH


def compute_haar_psi(img1, img2, n_levels=3, c=30.0):
    """
    Haar wavelet-based Perceptual Similarity Index (HaarPSI).

    Simplified version of Reisenhofer et al. 2018.  Compares Haar
    wavelet coefficients across scales — captures structural similarity
    in a way that correlates well with human perception.

    Parameters
    ----------
    img1, img2 : (H, W) arrays, assumed [0, 255] scale internally.
    n_levels : int, number of wavelet decomposition levels.
    c : float, stabilisation constant.

    Returns
    -------
    float in [0, 1].  1 = identical.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Scale to [0, 255] if input is [0, 1]
    if img1.max() <= 1.0 + 1e-6:
        img1 = img1 * 255.0
        img2 = img2 * 255.0

    a1, a2 = img1, img2
    local_sim_maps = []
    weights_maps = []

    for level in range(n_levels):
        LL1, LH1, HL1, HH1 = _haar_wavelet_2d(a1)
        LL2, LH2, HL2, HH2 = _haar_wavelet_2d(a2)

        # Directional coefficients to compare
        for (c1, c2) in [(LH1, LH2), (HL1, HL2)]:
            # Local similarity: 2|c1||c2| + C  /  |c1|² + |c2|² + C
            abs1, abs2 = np.abs(c1), np.abs(c2)
            sim = (2 * abs1 * abs2 + c) / (abs1 ** 2 + abs2 ** 2 + c)
            local_sim_maps.append(sim)

            # Weight = max magnitude (important regions contribute more)
            w = np.maximum(abs1, abs2)
            weights_maps.append(w)

        a1, a2 = LL1, LL2

    # Aggregate: weighted mean of logit-transformed similarities
    numerator = 0.0
    denominator = 0.0
    for sim_map, w_map in zip(local_sim_maps, weights_maps):
        # logit transform: log(s / (1 - s)) — maps [0,1] → ℝ
        sim_clamped = np.clip(sim_map, 1e-10, 1.0 - 1e-10)
        logit_sim = np.log(sim_clamped / (1.0 - sim_clamped))

        numerator += (logit_sim * w_map).sum()
        denominator += w_map.sum()

    if denominator < 1e-10:
        return 1.0

    # Inverse logit of the weighted mean
    avg_logit = numerator / denominator
    haar_psi = 1.0 / (1.0 + np.exp(-avg_logit))

    return float(haar_psi ** 2)  # Squared for sharper discrimination


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience: run all metrics at once
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_metrics(warped, target, displacement, source=None,
                        dsc_threshold=0.5, data_range=1.0):
    """
    Compute a full suite of evaluation metrics.

    Parameters
    ----------
    warped : (H, W) — warped source image.
    target : (H, W) — fixed target image.
    displacement : (2, H, W) or (H, W, 2) — displacement field.
    source : (H, W), optional — original source (for "before" metrics).
    dsc_threshold : float — binarisation threshold for DSC.
    data_range : float — dynamic range for SSIM/MS-SSIM.

    Returns
    -------
    dict of metric names → values.
    """
    results = {}

    # ---- similarity after registration ----
    results['ncc']     = compute_ncc(warped, target)
    results['mse']     = compute_mse(warped, target)
    results['ssim']    = compute_ssim(warped, target, data_range=data_range)
    results['ms_ssim'] = compute_ms_ssim(warped, target, data_range=data_range)
    results['dsc']     = compute_dsc(warped, target, threshold=dsc_threshold)
    results['haar_psi'] = compute_haar_psi(warped, target)

    # ---- Jacobian analysis ----
    _, jac_stats = jacobian_determinant_2d(displacement)
    results['jac_mean']    = jac_stats['mean']
    results['jac_std']     = jac_stats['std']
    results['jac_min']     = jac_stats['min']
    results['jac_pct_neg'] = jac_stats['pct_neg']

    # ---- similarity before registration (if source provided) ----
    if source is not None:
        results['ncc_before'] = compute_ncc(source, target)
        results['mse_before'] = compute_mse(source, target)

    return results
