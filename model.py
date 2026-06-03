"""
Attention 3D U-Net with Residual Blocks and Deep Supervision.

Improvements over vanilla 3D U-Net:
  - Residual connections inside every double-conv block
  - Attention gates in the decoder (suppress irrelevant skip features)
  - Deep supervision heads on the 3 intermediate decoder stages
  - Kaiming weight initialisation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualConvBlock(nn.Module):
    """Two 3x3x3 convolutions + residual shortcut projection."""

    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv1    = nn.Conv3d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1      = nn.BatchNorm3d(out_ch)
        self.conv2    = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm3d(out_ch)
        self.drop     = nn.Dropout3d(dropout)
        self.relu     = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_ch)
            ) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        skip = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return self.relu(x + skip)


class AttentionGate(nn.Module):
    """
    Soft attention gate.
    Learns a spatial attention map that scales the encoder skip connection
    so the decoder focuses only on relevant tumour regions.

    g : gating signal  (from decoder, after upsampling)
    x : skip connection (from encoder, same spatial size as g)
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(
            nn.Conv3d(F_g,   F_int, kernel_size=1, bias=False),
            nn.BatchNorm3d(F_int)
        )
        self.W_x  = nn.Sequential(
            nn.Conv3d(F_l,   F_int, kernel_size=1, bias=False),
            nn.BatchNorm3d(F_int)
        )
        self.psi  = nn.Sequential(
            nn.Conv3d(F_int, 1,     kernel_size=1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        alpha = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * alpha   # spatially weighted skip connection


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class AttentionUNet3D(nn.Module):
    """
    Attention 3D U-Net.

    Parameters
    ----------
    in_channels  : number of input MRI modalities  (4 for BraTS)
    out_channels : number of segmentation classes   (4 for BraTS 2020)
    features     : channel counts at each encoder level + bottleneck
                   e.g. (16, 32, 64, 128, 256)
    """

    def __init__(self, in_channels=4, out_channels=4,
                 features=(16, 32, 64, 128, 256)):
        super().__init__()
        features = list(features)

        # ---- Encoder -------------------------------------------------------
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        prev = in_channels
        for f in features[:-1]:                      # skip bottleneck channel
            self.encoders.append(ResidualConvBlock(prev, f, dropout=0.1))
            self.pools.append(nn.MaxPool3d(kernel_size=2))
            prev = f

        # ---- Bottleneck ----------------------------------------------------
        self.bottleneck = ResidualConvBlock(features[-2], features[-1], dropout=0.2)

        # ---- Decoder -------------------------------------------------------
        # rev_features = [128, 64, 32, 16]  (for default feature sizes)
        rev_features    = list(reversed(features[:-1]))
        self.upconvs    = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        self.decoders   = nn.ModuleList()

        prev_dec = features[-1]
        for f in rev_features:
            self.upconvs.append(
                nn.ConvTranspose3d(prev_dec, f, kernel_size=2, stride=2)
            )
            self.attn_gates.append(AttentionGate(F_g=f, F_l=f, F_int=f // 2))
            self.decoders.append(ResidualConvBlock(f * 2, f, dropout=0.1))
            prev_dec = f

        # ---- Final output head ---------------------------------------------
        self.output_conv = nn.Conv3d(rev_features[-1], out_channels, kernel_size=1)

        # ---- Deep supervision heads (all decoder stages except the last) ---
        # During training these produce outputs at 1/8, 1/4, 1/2 resolution.
        self.deep_sup = nn.ModuleList([
            nn.Conv3d(f, out_channels, kernel_size=1)
            for f in rev_features[:-1]          # all stages except final
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d,)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(self, x):
        # Encoder path
        enc_feats = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            enc_feats.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        enc_feats = list(reversed(enc_feats))
        deep_outs = []

        for i, (up, attn, dec) in enumerate(
                zip(self.upconvs, self.attn_gates, self.decoders)):
            x    = up(x)
            skip = attn(g=x, x=enc_feats[i])
            x    = dec(torch.cat([x, skip], dim=1))

            if i < len(self.deep_sup):
                deep_outs.append(self.deep_sup[i](x))

        main_out = self.output_conv(x)

        # During training return main output + list of auxiliary outputs.
        # During eval return only the main output (no overhead).
        if self.training:
            return main_out, deep_outs
        return main_out


def count_parameters(model):
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
