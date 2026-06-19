"""
Plain 3D U-Net — baseline model.

Perbedaan dari Attention 3D U-Net (model.py):
  - ConvBlock biasa (tanpa residual shortcut)
  - Tanpa Attention Gate di decoder
  - Tanpa Deep Supervision
  - Forward selalu return single tensor
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3x3 convolutions with BN + ReLU. No residual connection."""

    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """
    Plain 3D U-Net.

    Parameters
    ----------
    in_channels  : number of input MRI modalities  (4 for BraTS)
    out_channels : number of segmentation classes   (4 for BraTS 2020)
    features     : channel counts at each encoder level + bottleneck
    """

    def __init__(self, in_channels=4, out_channels=4,
                 features=(16, 32, 64, 128, 256)):
        super().__init__()
        features = list(features)

        # ---- Encoder -------------------------------------------------------
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        prev = in_channels
        for f in features[:-1]:
            self.encoders.append(ConvBlock(prev, f, dropout=0.1))
            self.pools.append(nn.MaxPool3d(kernel_size=2))
            prev = f

        # ---- Bottleneck ----------------------------------------------------
        self.bottleneck = ConvBlock(features[-2], features[-1], dropout=0.2)

        # ---- Decoder -------------------------------------------------------
        rev_features = list(reversed(features[:-1]))
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_dec = features[-1]
        for f in rev_features:
            self.upconvs.append(
                nn.ConvTranspose3d(prev_dec, f, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(f * 2, f, dropout=0.1))
            prev_dec = f

        # ---- Output head ---------------------------------------------------
        self.output_conv = nn.Conv3d(rev_features[-1], out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(self, x):
        # Encoder
        enc_feats = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            enc_feats.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder — plain concatenate, no attention gate
        enc_feats = list(reversed(enc_feats))
        for up, dec, skip in zip(self.upconvs, self.decoders, enc_feats):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))

        return self.output_conv(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
