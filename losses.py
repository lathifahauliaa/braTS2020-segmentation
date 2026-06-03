"""
Loss functions for BraTS 2020 brain tumour segmentation.

CombinedLoss = TverskyLoss + 0.5 x FocalLoss
  - TverskyLoss  : penalises false negatives more than false positives
                   (alpha=0.3 FP weight, beta=0.7 FN weight)
  - FocalLoss    : down-weights easy background voxels
Background class (index 0) is excluded from Tversky averaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, num_classes=4, alpha=0.3, beta=0.7, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.alpha  = alpha   # false positive weight
        self.beta   = beta    # false negative weight (higher = miss less tumour)
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.num_classes).permute(0, 4, 1, 2, 3).float()

        tversky_sum = 0.0
        for c in range(1, self.num_classes):    # skip background (class 0)
            p  = probs[:, c].contiguous().view(-1)
            t  = one_hot[:, c].contiguous().view(-1)
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            tversky_sum += (tp + self.smooth) / (
                tp + self.alpha * fp + self.beta * fn + self.smooth)

        return 1.0 - tversky_sum / (self.num_classes - 1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, reduction='none')
        pt   = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


class CombinedLoss(nn.Module):
    """Primary training objective: Tversky + 0.5 x Focal."""
    def __init__(self, num_classes=4):
        super().__init__()
        self.tversky = TverskyLoss(num_classes=num_classes, alpha=0.3, beta=0.7)
        self.focal   = FocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, logits, targets):
        return self.tversky(logits, targets) + 0.5 * self.focal(logits, targets)
