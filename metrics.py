"""
BraTS 2020 evaluation metrics.

Three overlapping tumour sub-regions (official BraTS challenge):
  WT  Whole Tumour     = labels 1 + 2 + 3
  TC  Tumour Core      = labels 1 + 3        (necrosis + enhancing)
  ET  Enhancing Tumour = label  3

Dice is computed per region, averaged over the batch.
"""

import torch
import numpy as np


def _region_mask(label_map, labels):
    mask = torch.zeros_like(label_map, dtype=torch.bool)
    for l in labels:
        mask |= (label_map == l)
    return mask


def dice_per_region(pred_logits, targets, smooth=1e-5):
    """
    Returns dict {'WT': float, 'TC': float, 'ET': float}.

    pred_logits : (B, C, H, W, D)  raw logits
    targets     : (B, H, W, D)     ground-truth class indices
    """
    pred_labels = pred_logits.argmax(dim=1)

    regions = {
        'WT': [1, 2, 3],
        'TC': [1, 3],
        'ET': [3],
    }

    scores = {}
    for name, labels in regions.items():
        p = _region_mask(pred_labels, labels).float().view(-1)
        t = _region_mask(targets,     labels).float().view(-1)
        inter = (p * t).sum()
        dice  = (2 * inter + smooth) / (p.sum() + t.sum() + smooth)
        scores[name] = dice.item()

    return scores


def iou_per_class(pred_logits, targets, num_classes=4, smooth=1e-5):
    """Mean IoU across foreground classes (1, 2, 3)."""
    pred_labels = pred_logits.argmax(dim=1)
    ious = []
    for c in range(1, num_classes):
        p     = (pred_labels == c).float().view(-1)
        t     = (targets     == c).float().view(-1)
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        ious.append(((inter + smooth) / (union + smooth)).item())
    return float(np.mean(ious))
