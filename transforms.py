"""
3D volumetric augmentation transforms.
Each transform takes (image: Tensor[4,H,W,D], mask: Tensor[H,W,D])
and returns the augmented pair.
"""

import random
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomFlip3D:
    """Randomly flip along any spatial axis."""
    def __init__(self, axes=(0, 1, 2), p=0.5):
        self.axes = axes
        self.p    = p

    def __call__(self, image, mask):
        for ax in self.axes:
            if random.random() < self.p:
                image = torch.flip(image, [ax + 1])  # +1 skips channel dim
                mask  = torch.flip(mask,  [ax])
        return image, mask


class RandomRotate90_3D:
    """Randomly rotate 90 / 180 / 270 degrees on a random spatial plane."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            k    = random.choice([1, 2, 3])
            axes = random.choice([(1, 2), (1, 3), (2, 3)])
            image = torch.rot90(image, k, list(axes))
            # mask has no channel dim, so subtract 1 from each axis index
            mask  = torch.rot90(mask,  k, [a - 1 for a in axes])
        return image, mask


class RandomIntensityShift:
    """Per-channel random brightness and contrast perturbation."""
    def __init__(self, shift=0.1, scale=0.1, p=0.5):
        self.shift = shift
        self.scale = scale
        self.p     = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            shift = (torch.rand(4, 1, 1, 1) - 0.5) * 2 * self.shift
            scale = 1.0 + (torch.rand(4, 1, 1, 1) - 0.5) * 2 * self.scale
            image = image * scale + shift
        return image, mask


class GaussianNoise:
    """Add zero-mean Gaussian noise."""
    def __init__(self, std=0.05, p=0.3):
        self.std = std
        self.p   = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = image + torch.randn_like(image) * self.std
        return image, mask


class RandomDropModality:
    """Zero out one random MRI channel — trains robustness to missing scans."""
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            ch = random.randint(0, image.shape[0] - 1)
            image = image.clone()
            image[ch] = 0.0
        return image, mask


def get_train_transforms():
    return Compose([
        RandomFlip3D(axes=(0, 1, 2), p=0.5),
        RandomRotate90_3D(p=0.3),
        RandomIntensityShift(shift=0.1, scale=0.1, p=0.5),
        GaussianNoise(std=0.05, p=0.3),
        RandomDropModality(p=0.1),
    ])
