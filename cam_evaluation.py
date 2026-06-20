"""
BraTS 2020 — CAM Evaluation Metrics
Metrics : DAUC (Deletion AUC) | IAUC (Insertion AUC) | IoU

Uses pytorch-grad-cam (github.com/jacobgil/pytorch-grad-cam) with a custom
3D segmentation target for volumetric (H, W, D) spatial dimensions.

References:
  - Samek et al. (2017), "Evaluating the Visualization of What a Deep Neural Network has Learned"
  - Petsiuk et al. (2018), RISE: Randomized Input Sampling for Explanation of Black-box Models

Run:
    python cam_evaluation.py --model_type attention   # Attention 3D U-Net (default)
    python cam_evaluation.py --model_type unet3d      # Plain 3D U-Net
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
import pandas as pd

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM, AblationCAM

# =============================================================================
# CONFIG
# =============================================================================
IS_KAGGLE = os.path.exists('/kaggle/working')
IS_COLAB  = os.path.exists('/content') and not IS_KAGGLE

if IS_KAGGLE:
    DATA_ROOT      = '/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    CKPT_ATTENTION = '/kaggle/working/checkpoints/best_model.pth'
    CKPT_UNET3D    = '/kaggle/working/checkpoints_unet3d/best_model_unet3d.pth'
    OUT_ATTENTION  = '/kaggle/working/cam_eval_attention'
    OUT_UNET3D     = '/kaggle/working/cam_eval_unet3d'
elif IS_COLAB:
    DRIVE_ROOT     = '/content/drive/MyDrive/skripsi'
    DATA_ROOT      = f'{DRIVE_ROOT}/dataset/brats/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    CKPT_ATTENTION = f'{DRIVE_ROOT}/checkpoints/best_model.pth'
    CKPT_UNET3D    = f'{DRIVE_ROOT}/checkpoints_unet3d/best_model_unet3d.pth'
    OUT_ATTENTION  = f'{DRIVE_ROOT}/cam_eval_attention'
    OUT_UNET3D     = f'{DRIVE_ROOT}/cam_eval_unet3d'
else:
    DATA_ROOT      = r'D:\skripsi\dataset\brats\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
    CKPT_ATTENTION = r'D:\skripsi\checkpoints\best_model.pth'
    CKPT_UNET3D    = r'D:\skripsi\checkpoints_unet3d\best_model_unet3d.pth'
    OUT_ATTENTION  = r'D:\skripsi\cam_eval_attention'
    OUT_UNET3D     = r'D:\skripsi\cam_eval_unet3d'

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CROP_SIZE   = (128, 128, 128)
N_PATIENTS  = None
SEED        = 42
N_STEPS     = 10
CLASS_NAMES = {1: 'Necrotic', 2: 'Oedema', 3: 'Enhancing'}
MODALITIES  = ['flair', 't1', 't1ce', 't2']

CAM_CLASSES = {
    'GradCAM'    : GradCAM,
    'GradCAM++'  : GradCAMPlusPlus,
    'ScoreCAM'   : ScoreCAM,
    'XGradCAM'   : XGradCAM,
    'AblationCAM': AblationCAM,
}


# =============================================================================
# DATA LOADING
# =============================================================================
def load_nii(path):
    return np.array(nib.load(path).dataobj).astype(np.float32)

def zscore(vol):
    mask = vol > 0
    if mask.sum() > 0:
        vol = np.where(mask, (vol - vol[mask].mean()) / (vol[mask].std() + 1e-8), 0.0)
    return vol

def crop_center(arr, crop=(128, 128, 128)):
    h, w, d    = arr.shape[-3], arr.shape[-2], arr.shape[-1]
    sh, sw, sd = (h-crop[0])//2, (w-crop[1])//2, (d-crop[2])//2
    if arr.ndim == 4:
        return arr[:, sh:sh+crop[0], sw:sw+crop[1], sd:sd+crop[2]]
    return arr[sh:sh+crop[0], sw:sw+crop[1], sd:sd+crop[2]]

def remap_labels(seg):
    out = np.zeros_like(seg, dtype=np.int64)
    out[seg == 1] = 1; out[seg == 2] = 2; out[seg == 4] = 3
    return out

def load_patient(patient_dir):
    vols = []
    for mod in MODALITIES:
        path = glob.glob(os.path.join(patient_dir, f'*{mod}*.nii*'))[0]
        vols.append(zscore(load_nii(path)))
    image    = crop_center(np.stack(vols, axis=0), CROP_SIZE)
    seg_path = glob.glob(os.path.join(patient_dir, '*seg*.nii*'))[0]
    seg      = crop_center(remap_labels(load_nii(seg_path)), CROP_SIZE)
    return torch.from_numpy(image).float().unsqueeze(0), seg


# =============================================================================
# CUSTOM TARGET — 3D Semantic Segmentation
# Extends pytorch-grad-cam's SemanticSegmentationTarget for volumetric data.
# Reference: github.com/jacobgil/pytorch-grad-cam
# =============================================================================
class SemanticSegmentationTarget3D:
    """
    CAM target for 3D segmentation.
    Equivalent to pytorch-grad-cam SemanticSegmentationTarget but for (H, W, D) spatial dims.
    """
    def __init__(self, category, pred_mask):
        self.category  = category
        self.pred_mask = (torch.from_numpy(pred_mask.astype(np.float32))
                         if isinstance(pred_mask, np.ndarray)
                         else pred_mask.float())

    def __call__(self, model_output):
        # model_output inside pytorch-grad-cam: (C, H, W, D) — no batch dim
        if self.pred_mask.sum() > 0:
            return (model_output[self.category] *
                    self.pred_mask.to(model_output.device)).sum()
        return model_output[self.category].sum()


# =============================================================================
# CAM COMPUTATION
# =============================================================================
def compute_cam(model, target_layer, image_t, class_idx):
    """Compute a single CAM heatmap (3D) using pytorch-grad-cam."""
    with torch.no_grad():
        out = model(image_t)
    pred_mask = (out.squeeze(0).argmax(dim=0) == class_idx).cpu().numpy()
    targets   = [SemanticSegmentationTarget3D(class_idx, pred_mask)]
    return targets


def run_cam(model, target_layer, image_t, class_idx, CAMClass):
    targets = compute_cam(model, target_layer, image_t, class_idx)
    with CAMClass(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=image_t, targets=targets)
    cam_3d = grayscale_cam[0]                    # (H, W, D)
    cam_3d = np.maximum(cam_3d, 0)
    cam_3d = gaussian_filter(cam_3d, sigma=2.0)
    if cam_3d.max() > 0:
        cam_3d = (cam_3d - cam_3d.min()) / (cam_3d.max() - cam_3d.min() + 1e-8)
    return cam_3d


# =============================================================================
# METRIC FUNCTIONS
# Reference: Samek et al. (2017) — Evaluating the Visualization of What a DNN has Learned
# =============================================================================
def _model_score(model, x, class_idx):
    with torch.no_grad():
        out = model(x)
    import torch.nn.functional as F
    return torch.softmax(out, dim=1)[:, class_idx].mean().item()


def compute_dauc(model, x, cam, class_idx, n_steps=N_STEPS):
    """Deletion AUC — lower is better. Deletes most important voxels first."""
    H, W, D = cam.shape
    order   = np.argsort(cam.ravel())[::-1]
    total   = len(order)
    scores  = [_model_score(model, x, class_idx)]

    for s in range(1, n_steps + 1):
        k      = int(s / n_steps * total)
        mask   = np.ones(total, dtype=np.float32)
        mask[order[:k]] = 0
        m3d    = torch.from_numpy(mask.reshape(H, W, D)).to(x.device)
        scores.append(_model_score(model, x * m3d.unsqueeze(0).unsqueeze(0), class_idx))

    return float(np.trapezoid(scores, np.linspace(0, 1, n_steps + 1))
                 if hasattr(np, 'trapezoid')
                 else np.trapz(scores, np.linspace(0, 1, n_steps + 1)))


def compute_iauc(model, x, cam, class_idx, n_steps=N_STEPS):
    """Insertion AUC — higher is better. Inserts most important voxels into blurred baseline."""
    f_full = _model_score(model, x, class_idx)
    if f_full < 1e-8:
        return 0.0

    x_np   = x.squeeze(0).cpu().numpy()
    x_blur = torch.from_numpy(
        np.stack([gaussian_filter(x_np[c], sigma=10) for c in range(x_np.shape[0])], axis=0)
    ).unsqueeze(0).to(x.device)

    H, W, D = cam.shape
    order   = np.argsort(cam.ravel())[::-1]
    total   = len(order)
    scores  = [min(_model_score(model, x_blur, class_idx) / f_full, 1.0)]

    for s in range(1, n_steps + 1):
        k     = int(s / n_steps * total)
        mask  = np.zeros(total, dtype=np.float32)
        mask[order[:k]] = 1
        m3d   = torch.from_numpy(mask.reshape(H, W, D)).to(x.device)
        x_ins = x_blur + m3d.unsqueeze(0).unsqueeze(0) * (x - x_blur)
        scores.append(min(_model_score(model, x_ins, class_idx) / f_full, 1.0))

    return float(np.trapezoid(scores, np.linspace(0, 1, n_steps + 1))
                 if hasattr(np, 'trapezoid')
                 else np.trapz(scores, np.linspace(0, 1, n_steps + 1)))


def compute_iou(cam, gt_mask, threshold=0.5):
    """IoU between thresholded CAM and GT binary mask — higher is better."""
    cam_bin = (cam >= threshold)
    gt_bin  = gt_mask.astype(bool)
    inter   = (cam_bin & gt_bin).sum()
    union   = (cam_bin | gt_bin).sum()
    return float(inter / (union + 1e-8))


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['attention', 'unet3d'], default='attention',
                        help='Model architecture: attention (default) | unet3d')
    args = parser.parse_args()

    if args.model_type == 'attention':
        CKPT_PATH  = CKPT_ATTENTION
        OUTPUT_DIR = OUT_ATTENTION
        arch_label = "Attention 3D U-Net"
    else:
        CKPT_PATH  = CKPT_UNET3D
        OUTPUT_DIR = OUT_UNET3D
        arch_label = "Plain 3D U-Net"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print(f"  BraTS 2020 — CAM Evaluation: DAUC | IAUC | IoU")
    print(f"  Architecture : {arch_label}")
    print("  (via pytorch-grad-cam — github.com/jacobgil/pytorch-grad-cam)")
    print("=" * 65)

    if args.model_type == 'attention':
        from model import AttentionUNet3D
        model      = AttentionUNet3D(in_channels=4, out_channels=4).to(DEVICE)
        get_target = lambda m: m.decoders[1].conv2
    else:
        from model_unet3d import UNet3D
        model      = UNet3D(in_channels=4, out_channels=4).to(DEVICE)
        get_target = lambda m: m.decoders[1].block[4]

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Model loaded — Epoch {ckpt.get('epoch','?')}, "
          f"Best Dice {ckpt.get('best_mean_dice', 0):.4f}")

    target_layer = get_target(model)

    all_patients = sorted(glob.glob(os.path.join(DATA_ROOT, 'BraTS20_Training_*')))[:N_PATIENTS]
    _, test_patients = train_test_split(all_patients, test_size=0.09, random_state=SEED)
    print(f"  Test set: {len(test_patients)} pasien\n")

    records = []

    for p_idx, pdir in enumerate(test_patients, 1):
        pid = os.path.basename(pdir)
        print(f"[{p_idx}/{len(test_patients)}] {pid}")

        image_t, gt = load_patient(pdir)
        image_t = image_t.to(DEVICE)

        for class_idx, class_name in CLASS_NAMES.items():
            gt_mask = (gt == class_idx)
            if gt_mask.sum() == 0:
                print(f"  Skipping {class_name} — no GT voxels")
                continue

            print(f"  Class: {class_name}")

            for method_name, CAMClass in CAM_CLASSES.items():
                print(f"    {method_name}...", end=' ', flush=True)

                cam = run_cam(model, target_layer, image_t, class_idx, CAMClass)

                dauc = compute_dauc(model, image_t, cam, class_idx)
                iauc = compute_iauc(model, image_t, cam, class_idx)
                iou  = compute_iou(cam, gt_mask)

                print(f"DAUC={dauc:.4f}  IAUC={iauc:.4f}  IoU={iou:.4f}")

                records.append({
                    'Patient'  : pid,
                    'Class'    : class_name,
                    'Method'   : method_name,
                    'DAUC'     : round(dauc, 4),
                    'IAUC'     : round(iauc, 4),
                    'IoU'      : round(iou, 4),
                })

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, f'cam_metrics_{args.model_type}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    print("\n" + "=" * 65)
    print("  SUMMARY — Mean per Method")
    print("=" * 65)
    summary = df.groupby(['Method', 'Class'])[['DAUC', 'IAUC', 'IoU']].mean().round(4)
    print(summary.to_string())

    print("\n" + "=" * 65)
    print("  SUMMARY — Mean per Method (all classes)")
    print("=" * 65)
    overall = df.groupby('Method')[['DAUC', 'IAUC', 'IoU']].mean().round(4)
    overall['DAUC_rank'] = overall['DAUC'].rank()
    overall['IAUC_rank'] = overall['IAUC'].rank(ascending=False)
    overall['IoU_rank']  = overall['IoU'].rank(ascending=False)
    print(overall.to_string())
