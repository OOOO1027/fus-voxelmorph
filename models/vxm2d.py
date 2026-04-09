"""
2D VoxelMorph registration network for functional ultrasound (fUS) imaging.

Architecture follows VoxelMorph's VxmDense / VxmPairwise design:
  - UNet backbone predicts a velocity (or displacement) field
  - Optional scaling-and-squaring integration for diffeomorphic registration
  - Differentiable spatial transformer warps the moving image

Designed for fUS power Doppler maps with typical FOV 128 x 100 (non-square).

References
----------
VoxelMorph: A Learning Framework for Deformable Medical Image Registration.
G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca.
IEEE TMI, 38(8), pp 1788-1800, 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Building blocks
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    轻量级通道注意力，帮助模型关注血管结构等信息丰富的特征通道。

    Parameters
    ----------
    channels : int
    reduction : int
        通道压缩比，默认 4
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ConvBlock(nn.Module):
    """
    2D convolution + normalization + activation + optional residual + optional SE.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    stride : int
        Use stride=2 for downsampling in the encoder.
    norm : str
        Normalization type: 'instance' | 'batch' | 'none'.
        InstanceNorm is preferred for small-batch medical imaging.
    activation : str
        'leakyrelu' (default) or 'relu'.
    residual : bool
        If True and in_channels == out_channels and stride == 1,
        add a residual skip connection.
    se_attention : bool
        If True, add Squeeze-and-Excitation channel attention.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 norm='instance', activation='leakyrelu',
                 residual=False, se_attention=False):
        super().__init__()
        padding = kernel_size // 2

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding)]

        if norm == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == 'instance':
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        # 'none' → no normalization layer

        if activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

        # Residual connection: only when dimensions match
        self.use_residual = residual and (in_channels == out_channels) and (stride == 1)

        # SE attention
        self.se = SEBlock(out_channels) if se_attention else None

    def forward(self, x):
        out = self.block(x)
        if self.se is not None:
            out = self.se(out)
        if self.use_residual:
            out = out + x
        return out


# ---------------------------------------------------------------------------
#  UNet backbone
# ---------------------------------------------------------------------------

class Unet(nn.Module):
    """
    2D UNet backbone following VoxelMorph's architecture.

    Encoder uses stride-2 convolutions for downsampling.
    Decoder uses bilinear upsampling + skip concatenation + convolution.

    The decoder may have **more** entries than the encoder: the first
    ``len(enc_channels)`` decoder layers correspond to upsampling levels
    (each with a skip connection from the encoder), and any remaining
    decoder layers are extra convolutions at full resolution — exactly
    matching VoxelMorph's default layout::

        enc_channels = [16, 32, 32, 32]          # 4 downsampling levels
        dec_channels = [32, 32, 32, 32, 16, 16]  # 4 upsample + 2 full-res

    Non-square spatial sizes (e.g. 128 x 100) are handled by upsampling to
    the *exact* size of the corresponding skip tensor, avoiding the ×2
    rounding mismatch.

    Parameters
    ----------
    in_channels : int
        Number of input channels (source_ch + target_ch after concat).
    enc_channels : tuple of int
        Feature channels at each encoder level.
    dec_channels : tuple of int
        Feature channels at each decoder level.  len >= len(enc_channels).
    norm : str
        'instance' | 'batch' | 'none'.
    """

    def __init__(self, in_channels, enc_channels=(16, 32, 32, 32),
                 dec_channels=(32, 32, 32, 32, 16, 16), norm='instance'):
        super().__init__()
        n_enc = len(enc_channels)
        assert len(dec_channels) >= n_enc, \
            'dec_channels must have at least as many entries as enc_channels'

        # ---- encoder ----
        self.encoders = nn.ModuleList()
        ch = in_channels
        for enc_ch in enc_channels:
            self.encoders.append(ConvBlock(ch, enc_ch, stride=2, norm=norm))
            ch = enc_ch

        # ---- decoder (upsampling levels with skip connections) ----
        self.up_convs = nn.ModuleList()
        enc_out_channels = [in_channels] + list(enc_channels)  # includes input
        for i in range(n_enc):
            skip_ch = enc_out_channels[n_enc - 1 - i]  # matching encoder output
            self.up_convs.append(ConvBlock(
                ch + skip_ch, dec_channels[i], norm=norm,
                se_attention=True,  # 解码器使用 SE 注意力
            ))
            ch = dec_channels[i]

        # ---- extra full-resolution convolutions (no skip, no upsample) ----
        self.extra_convs = nn.ModuleList()
        for i in range(n_enc, len(dec_channels)):
            self.extra_convs.append(ConvBlock(
                ch, dec_channels[i], norm=norm,
                residual=True,  # 全分辨率层使用残差连接
            ))
            ch = dec_channels[i]

        self.out_channels = ch

    def forward(self, x):
        # ---- encoder ----
        skips = [x]                     # skips[0] = original input
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)             # skips[1..n_enc] = encoder outputs

        # ---- decoder (upsampling) ----
        # x is currently skips[-1] (deepest encoder output / bottleneck)
        for i, up_conv in enumerate(self.up_convs):
            skip = skips[-(i + 2)]      # go backwards through skips
            # Upsample to the exact spatial size of the skip tensor
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = up_conv(x)

        # ---- extra full-res convolutions ----
        for conv in self.extra_convs:
            x = conv(x)

        return x


# ---------------------------------------------------------------------------
#  Spatial Transformer
# ---------------------------------------------------------------------------

class SpatialTransformer(nn.Module):
    """
    2D differentiable spatial transformer.

    Warps a *moving* image according to a displacement field using
    ``F.grid_sample`` with bilinear interpolation.

    The identity grid is cached as a persistent buffer so it is created only
    once per spatial size (and moves to the correct device automatically).

    Parameters
    ----------
    size : tuple of int, optional
        Spatial size (H, W) to pre-allocate the identity grid.  If ``None``,
        the grid is built lazily on the first forward call.
    """

    def __init__(self, size=None):
        super().__init__()
        self._cur_size = None
        if size is not None:
            self._build_grid(size)

    # ---- internal helpers --------------------------------------------------

    def _build_grid(self, size):
        """Create a normalized identity grid and register it as a buffer."""
        H, W = size
        grid_y = torch.linspace(-1.0, 1.0, H)
        grid_x = torch.linspace(-1.0, 1.0, W)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)  x-first for grid_sample
        # Register as buffer so it follows .to(device) / .half() etc.
        if hasattr(self, 'grid'):
            del self.grid
        self.register_buffer('grid', grid, persistent=False)
        self._cur_size = size

    def _ensure_grid(self, displacement):
        """Lazily (re-)build the grid when the spatial size changes."""
        size = tuple(displacement.shape[2:])        # (H, W)
        if size != self._cur_size:
            self._build_grid(size)
        # Always ensure grid is on the same device as displacement
        if self.grid.device != displacement.device:
            self.grid = self.grid.to(displacement.device)

    # ---- forward -----------------------------------------------------------

    def forward(self, moving, displacement):
        """
        Parameters
        ----------
        moving : Tensor, shape (B, C, H, W)
            Image(s) to warp.
        displacement : Tensor, shape (B, 2, H, W)
            Pixel-unit displacement field.  Channel 0 = dx (horizontal),
            channel 1 = dy (vertical).

        Returns
        -------
        Tensor, shape (B, C, H, W)
            Warped image(s).
        """
        self._ensure_grid(displacement)

        B, _, H, W = displacement.shape
        grid = self.grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

        # Convert pixel displacements to normalised [-1, 1] displacements.
        # grid_sample expects (x, y) order; displacement channels are (dx, dy).
        norm_disp = torch.stack([
            displacement[:, 0] * 2.0 / (W - 1),    # dx
            displacement[:, 1] * 2.0 / (H - 1),    # dy
        ], dim=-1)                                  # (B, H, W, 2)

        sample_grid = grid + norm_disp

        return F.grid_sample(
            moving, sample_grid,
            mode='bilinear', padding_mode='border', align_corners=True,
        )


# ---------------------------------------------------------------------------
#  Velocity-field integrator (scaling & squaring)
# ---------------------------------------------------------------------------

class VecInt(nn.Module):
    """
    Integrate a stationary velocity field via scaling and squaring.

    Produces a diffeomorphic displacement field from a velocity field,
    guaranteeing smooth and invertible transformations.

    Parameters
    ----------
    steps : int
        Number of squaring iterations.  The velocity is first scaled by
        1/(2^steps), then iteratively composed with itself.
    """

    def __init__(self, steps=7):
        super().__init__()
        assert steps >= 0
        self.steps = steps
        self.scale = 1.0 / (2 ** steps)
        self.transformer = SpatialTransformer()

    def forward(self, velocity):
        """
        Parameters
        ----------
        velocity : Tensor, shape (B, 2, H, W)

        Returns
        -------
        Tensor, shape (B, 2, H, W)
            Integrated displacement field.
        """
        disp = velocity * self.scale
        for _ in range(self.steps):
            disp = disp + self.transformer(disp, disp)
        return disp


# ---------------------------------------------------------------------------
#  Top-level registration network
# ---------------------------------------------------------------------------

class VxmDense2D(nn.Module):
    """
    VoxelMorph-style 2D dense registration network for fUS images.

    Architecture::

        source ─┐
                ├─ concat ─► UNet ─► flow_layer ─► velocity
        target ─┘                                      │
                                               [VecInt integration]
                                                       │
                                                  displacement
                                                       │
                                          SpatialTransformer(source)
                                                       │
                                                    warped

    Parameters
    ----------
    in_channels : int
        Channels per input image (1 for power Doppler).
    enc_channels : tuple of int
        Encoder feature widths.  Default ``(16, 32, 32, 32)``
        matches VoxelMorph.
    dec_channels : tuple of int
        Decoder feature widths.  Default ``(32, 32, 32, 32, 16, 16)``
        matches VoxelMorph (4 upsample + 2 full-res layers).
    norm : str
        Normalization: 'instance' (default, best for small batches),
        'batch', or 'none'.
    integration_steps : int
        Scaling-and-squaring steps for diffeomorphic mode.
        Set to 0 for direct displacement regression.
    bidir : bool
        If True, also compute the inverse displacement (target → source)
        by integrating -velocity. Useful for bidirectional training.
    """

    def __init__(
        self,
        in_channels=1,
        enc_channels=(16, 32, 32, 32),
        dec_channels=(32, 32, 32, 32, 16, 16),
        norm='instance',
        integration_steps=7,
        bidir=False,
    ):
        super().__init__()

        self.bidir = bidir
        self.integration_steps = integration_steps

        # UNet backbone — input is concat(source, target)
        self.unet = Unet(
            in_channels=in_channels * 2,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            norm=norm,
        )

        # Flow head: 3×3 conv from UNet features → 2-channel velocity/displacement.
        # Weights initialised near-zero so the initial predicted field ≈ identity,
        # which is critical for stable early training.
        self.flow = nn.Conv2d(self.unet.out_channels, 2, kernel_size=3, padding=1)
        nn.init.normal_(self.flow.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.flow.bias)

        # Spatial transformer for warping
        self.spatial_transformer = SpatialTransformer()

        # Optional diffeomorphic integration
        if integration_steps > 0:
            self.integrate = VecInt(steps=integration_steps)
        else:
            self.integrate = None

    def forward(self, source, target):
        """
        Parameters
        ----------
        source : Tensor (B, C, H, W) — moving image
        target : Tensor (B, C, H, W) — fixed image

        Returns
        -------
        warped_source : Tensor (B, C, H, W)
        pos_flow : Tensor (B, 2, H, W) — displacement field (source → target)

        If ``bidir=True``, additionally returns:
        warped_target : Tensor (B, C, H, W)
        neg_flow : Tensor (B, 2, H, W) — displacement field (target → source)
        """
        x = torch.cat([source, target], dim=1)       # (B, 2C, H, W)
        x = self.unet(x)                              # (B, dec[-1], H, W)
        velocity = self.flow(x)                        # (B, 2, H, W)

        # Integrate velocity → displacement
        if self.integrate is not None:
            pos_flow = self.integrate(velocity)
        else:
            pos_flow = velocity

        # Warp source image
        warped_source = self.spatial_transformer(source, pos_flow)

        if not self.bidir:
            return warped_source, pos_flow

        # Bidirectional: also warp target with the inverse flow
        neg_flow = self.integrate(-velocity) if self.integrate is not None else -velocity
        warped_target = self.spatial_transformer(target, neg_flow)

        return warped_source, pos_flow, warped_target, neg_flow
