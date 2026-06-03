"""
BraTS 2020 Dataset Loader

Folder structure expected:
  MICCAI_BraTS2020_TrainingData/
      BraTS20_Training_001/
          BraTS20_Training_001_flair.nii.gz
          BraTS20_Training_001_t1.nii.gz
          BraTS20_Training_001_t1ce.nii.gz
          BraTS20_Training_001_t2.nii.gz
          BraTS20_Training_001_seg.nii.gz
      BraTS20_Training_002/
      ...

Segmentation labels:
  0 = Background
  1 = Necrotic core / Non-enhancing tumour
  2 = Peritumoral oedema
  4 = Enhancing tumour  →  remapped to 3

Only the TRAINING split is used (validation has no seg masks).
We split training 80/20 inside the training script.
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

MODALITIES = ['flair', 't1', 't1ce', 't2']


class BraTS2020Dataset(Dataset):
    def __init__(self, patient_dirs, transform=None, crop_size=(128, 128, 128)):
        self.patient_dirs = patient_dirs
        self.transform    = transform
        self.crop_size    = crop_size

    def __len__(self):
        return len(self.patient_dirs)

    def _load_nii(self, path):
        try:
            img = nib.load(path)
            return np.array(img.dataobj).astype(np.float32)
        except Exception as e:
            print(f"\n[ERROR] Failed to load: {path}")
            print(f"[ERROR] Reason: {e}")
            raise

    def _zscore(self, vol):
        """Z-score normalise using only non-zero (brain) voxels."""
        mask = vol > 0
        if mask.sum() > 0:
            vol = np.where(mask,
                           (vol - vol[mask].mean()) / (vol[mask].std() + 1e-8),
                           0.0)
        return vol

    def _crop_center(self, arr, crop):
        """Centre-crop the last three spatial dimensions."""
        h, w, d = arr.shape[-3], arr.shape[-2], arr.shape[-1]
        ch, cw, cd = crop
        sh = (h - ch) // 2
        sw = (w - cw) // 2
        sd = (d - cd) // 2
        if arr.ndim == 4:
            return arr[:, sh:sh+ch, sw:sw+cw, sd:sd+cd]
        return arr[sh:sh+ch, sw:sw+cw, sd:sd+cd]

    def _remap_labels(self, seg):
        """Convert BraTS labels {0,1,2,4} to contiguous {0,1,2,3}."""
        out = np.zeros_like(seg, dtype=np.int64)
        out[seg == 1] = 1
        out[seg == 2] = 2
        out[seg == 4] = 3
        return out

    def __getitem__(self, idx):
        pdir = self.patient_dirs[idx]

        # Load and normalise each modality, then stack → (4, H, W, D)
        vols = []
        for mod in MODALITIES:
            path = glob.glob(os.path.join(pdir, f'*{mod}*.nii*'))[0]
            vols.append(self._zscore(self._load_nii(path)))
        image = np.stack(vols, axis=0)

        # Segmentation mask
        seg_path = glob.glob(os.path.join(pdir, '*seg*.nii*'))[0]
        seg = self._remap_labels(self._load_nii(seg_path))

        # Centre-crop to fixed spatial size
        image = self._crop_center(image, self.crop_size)
        seg   = self._crop_center(seg,   self.crop_size)

        image = torch.from_numpy(image).float()
        seg   = torch.from_numpy(seg).long()

        if self.transform:
            image, seg = self.transform(image, seg)

        return image, seg
