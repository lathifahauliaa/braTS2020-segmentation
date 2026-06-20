"""
BraTS 2020 - CAM Methods Comparison
Methods: GradCAM | GradCAM++ | ScoreCAM | XGradCAM | AblationCAM

Uses pytorch-grad-cam (github.com/jacobgil/pytorch-grad-cam) with a custom
3D segmentation target to support volumetric (H, W, D) spatial dimensions.

Run:
    python cam_visualization.py --model_type attention   # Attention 3D U-Net (default)
    python cam_visualization.py --model_type unet3d      # Plain 3D U-Net
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split

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
    OUT_ATTENTION  = '/kaggle/working/cam_results'
    OUT_UNET3D     = '/kaggle/working/cam_results_unet3d'
elif IS_COLAB:
    COLAB_DRIVE_ROOT = '/content/drive/MyDrive/skripsi'
    DATA_ROOT        = f'{COLAB_DRIVE_ROOT}/dataset/brats/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    CKPT_ATTENTION   = f'{COLAB_DRIVE_ROOT}/checkpoints/best_model.pth'
    CKPT_UNET3D      = f'{COLAB_DRIVE_ROOT}/checkpoints_unet3d/best_model_unet3d.pth'
    OUT_ATTENTION    = f'{COLAB_DRIVE_ROOT}/cam_results'
    OUT_UNET3D       = f'{COLAB_DRIVE_ROOT}/cam_results_unet3d'
else:
    DATA_ROOT      = r'D:\skripsi\dataset\brats\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
    CKPT_ATTENTION = r'D:\skripsi\checkpoints\best_model.pth'
    CKPT_UNET3D    = r'D:\skripsi\checkpoints_unet3d\best_model_unet3d.pth'
    OUT_ATTENTION  = r'D:\skripsi\cam_results'
    OUT_UNET3D     = r'D:\skripsi\cam_results_unet3d'

CROP_SIZE   = (128, 128, 128)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODALITIES  = ['flair', 't1', 't1ce', 't2']
CLASS_NAMES = {1: 'Necrotic_Core', 2: 'Oedema', 3: 'Enhancing_Tumour'}
SEED        = 42

CAM_CLASSES = {
    'GradCAM'    : GradCAM,
    'GradCAM++'  : GradCAMPlusPlus,
    'ScoreCAM'   : ScoreCAM,
    'XGradCAM'   : XGradCAM,
    'AblationCAM': AblationCAM,
}


# =============================================================================
# MODEL — Attention 3D U-Net
# =============================================================================
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv1    = nn.Conv3d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1      = nn.BatchNorm3d(out_ch)
        self.conv2    = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm3d(out_ch)
        self.drop     = nn.Dropout3d(dropout)
        self.relu     = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, bias=False), nn.BatchNorm3d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
    def forward(self, x):
        skip = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return self.relu(x + skip)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv3d(F_g,   F_int, 1, bias=False), nn.BatchNorm3d(F_int))
        self.W_x  = nn.Sequential(nn.Conv3d(F_l,   F_int, 1, bias=False), nn.BatchNorm3d(F_int))
        self.psi  = nn.Sequential(nn.Conv3d(F_int, 1,     1, bias=False), nn.BatchNorm3d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))

class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, features=(16,32,64,128,256)):
        super().__init__()
        features = list(features)
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        prev = in_channels
        for f in features[:-1]:
            self.encoders.append(ResidualConvBlock(prev, f))
            self.pools.append(nn.MaxPool3d(2))
            prev = f
        self.bottleneck = ResidualConvBlock(features[-2], features[-1], dropout=0.2)
        rev = list(reversed(features[:-1]))
        self.upconvs    = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        self.decoders   = nn.ModuleList()
        prev_dec = features[-1]
        for f in rev:
            self.upconvs.append(nn.ConvTranspose3d(prev_dec, f, 2, stride=2))
            self.attn_gates.append(AttentionGate(f, f, f // 2))
            self.decoders.append(ResidualConvBlock(f * 2, f))
            prev_dec = f
        self.output_conv = nn.Conv3d(rev[-1], out_channels, 1)
        self.deep_sup    = nn.ModuleList([nn.Conv3d(f, out_channels, 1) for f in rev[:-1]])

    def forward(self, x):
        enc_feats = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); enc_feats.append(x); x = pool(x)
        x = self.bottleneck(x)
        enc_feats = list(reversed(enc_feats))
        for i, (up, attn, dec) in enumerate(zip(self.upconvs, self.attn_gates, self.decoders)):
            x = up(x)
            x = dec(torch.cat([x, attn(g=x, x=enc_feats[i])], dim=1))
        return self.output_conv(x)


# =============================================================================
# MODEL — Plain 3D U-Net
# =============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, features=(16,32,64,128,256)):
        super().__init__()
        features = list(features)
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        prev = in_channels
        for f in features[:-1]:
            self.encoders.append(ConvBlock(prev, f))
            self.pools.append(nn.MaxPool3d(2))
            prev = f
        self.bottleneck = ConvBlock(features[-2], features[-1], dropout=0.2)
        rev = list(reversed(features[:-1]))
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_dec = features[-1]
        for f in rev:
            self.upconvs.append(nn.ConvTranspose3d(prev_dec, f, 2, stride=2))
            self.decoders.append(ConvBlock(f * 2, f))
            prev_dec = f
        self.output_conv = nn.Conv3d(rev[-1], out_channels, 1)

    def forward(self, x):
        enc_feats = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); enc_feats.append(x); x = pool(x)
        x = self.bottleneck(x)
        enc_feats = list(reversed(enc_feats))
        for up, dec, skip in zip(self.upconvs, self.decoders, enc_feats):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.output_conv(x)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_nii(path):
    import nibabel as nib
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
# Extends pytorch-grad-cam's SemanticSegmentationTarget for volumetric (H,W,D) data.
# Reference: github.com/jacobgil/pytorch-grad-cam
# =============================================================================
class SemanticSegmentationTarget3D:
    """
    CAM target for 3D segmentation.
    Scores = sum of softmax probabilities for `category` over predicted tumor voxels.
    Equivalent to pytorch-grad-cam SemanticSegmentationTarget but for 3D spatial dims.
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
# CAM COMPUTATION (using pytorch-grad-cam)
# =============================================================================
def compute_all_cams(model, target_layer, image_t, class_idx):
    """Run all 5 CAM methods via pytorch-grad-cam and return dict of 3D heatmaps."""

    # Prediction mask — fixed, non-differentiable
    with torch.no_grad():
        out = model(image_t)
    pred_mask = (out.squeeze(0).argmax(dim=0) == class_idx).cpu().numpy()  # (H,W,D) bool

    targets = [SemanticSegmentationTarget3D(class_idx, pred_mask)]
    hm      = {}

    for name, CAMClass in CAM_CLASSES.items():
        print(f"    [{list(CAM_CLASSES).index(name)+1}/5] {name}...", flush=True)
        with CAMClass(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=image_t, targets=targets)

        cam_3d = grayscale_cam[0]                    # (H, W, D)
        cam_3d = np.maximum(cam_3d, 0)
        cam_3d = gaussian_filter(cam_3d, sigma=2.0)  # smooth blocky artefacts
        if cam_3d.max() > 0:
            cam_3d = (cam_3d - cam_3d.min()) / (cam_3d.max() - cam_3d.min() + 1e-8)
        hm[name] = cam_3d

    return hm


# =============================================================================
# VISUALISATION — all 5 CAMs in one figure
# =============================================================================
def _colored_cam_slice(heatmaps_by_class, cam_name, pred_sl, sl):
    viridis_cmap = plt.cm.get_cmap('viridis')
    h, w         = pred_sl.shape
    rgba         = np.zeros((h, w, 4), dtype=np.float32)
    for class_idx in [1, 2, 3]:
        if class_idx not in heatmaps_by_class:
            continue
        cam_3d = heatmaps_by_class[class_idx].get(cam_name)
        if cam_3d is None:
            continue
        cam_sl = cam_3d[:, :, sl].copy()
        if cam_sl.max() > 0:
            cam_sl = (cam_sl - cam_sl.min()) / (cam_sl.max() - cam_sl.min() + 1e-8)
        mask = (pred_sl == class_idx)
        if not mask.any():
            continue
        color          = np.array(viridis_cmap(class_idx / 3.0))
        rgba[mask, :3] = color[:3]
        rgba[mask, 3]  = np.clip(cam_sl[mask] * 0.9, 0.15, 0.9)
    return rgba


def plot_cam_combined(image_np, gt, pred, heatmaps_by_class,
                      pid, slices, output_dir, arch_tag='attention'):
    cam_names = list(CAM_CLASSES.keys())
    n_cols    = 3 + len(cam_names)
    n_rows    = len(slices)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.8 * n_rows),
                             squeeze=False)

    col_titles = ['FLAIR', 'Ground Truth', 'Prediction'] + cam_names
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10, fontweight='bold', pad=8)

    for row, sl in enumerate(slices):
        flair = image_np[0, :, :, sl]
        gt_sl = gt[:, :, sl]
        pr_sl = pred[:, :, sl]

        axes[row, 0].set_ylabel(f'z = {sl}', fontsize=9)
        axes[row, 0].imshow(flair, cmap='gray', origin='lower')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(gt_sl, cmap='viridis', origin='lower', vmin=0, vmax=3)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(pr_sl, cmap='viridis', origin='lower', vmin=0, vmax=3)
        axes[row, 2].axis('off')

        viridis_cmap   = plt.cm.get_cmap('viridis')
        class_contours = {
            c: {
                'gt_mask': (gt_sl == c).astype(np.float32),
                'pr_mask': (pr_sl == c).astype(np.float32),
                'color':   viridis_cmap(c / 3.0)[:3],
            }
            for c in [1, 2, 3]
        }

        for c, info in class_contours.items():
            if info['gt_mask'].max() > 0:
                axes[row, 1].contour(info['gt_mask'], levels=[0.5],
                                     colors=[info['color']], linewidths=1.8, origin='lower')
        for c, info in class_contours.items():
            if info['pr_mask'].max() > 0:
                axes[row, 2].contour(info['pr_mask'], levels=[0.5],
                                     colors=[info['color']], linewidths=1.8, origin='lower')

        for col, name in enumerate(cam_names, start=3):
            axes[row, col].imshow(flair, cmap='gray', origin='lower')
            rgba = _colored_cam_slice(heatmaps_by_class, name, pr_sl, sl)
            axes[row, col].imshow(rgba, origin='lower')
            for c, info in class_contours.items():
                if info['pr_mask'].max() > 0:
                    axes[row, col].contour(info['pr_mask'], levels=[0.5],
                                           colors=[info['color']], linewidths=1.8, origin='lower')
            axes[row, col].axis('off')

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=3))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cb = fig.colorbar(sm, cax=cbar_ax, ticks=[0, 1, 2, 3])
    cb.set_ticklabels(['Background', 'Necrotic', 'Oedema', 'Enhancing'])
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(f'Patient: {pid}  —  CAM Comparison (All Classes)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.subplots_adjust(left=0.05, right=0.91, hspace=0.05, wspace=0.04)

    fname = os.path.join(output_dir, f'{pid}_CAM_{arch_tag}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved -> {fname}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_patients', type=int, default=None,
                        help='Jumlah pasien yang diproses (default: semua test set)')
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
    print(f"  BraTS 2020 — 3D CAM Comparison  [{arch_label}]")
    print("  GradCAM | GradCAM++ | ScoreCAM | XGradCAM | AblationCAM")
    print("  (via pytorch-grad-cam — github.com/jacobgil/pytorch-grad-cam)")
    print("=" * 65)

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"No checkpoint at:\n  {CKPT_PATH}")

    print("\nLoading model...")
    if args.model_type == 'attention':
        model = AttentionUNet3D(in_channels=4, out_channels=4).to(DEVICE)
    else:
        model = UNet3D(in_channels=4, out_channels=4).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    best = ckpt.get('best_dice', ckpt.get('best_mean_dice', 0))
    print(f"  Epoch={ckpt.get('epoch','?')}   Best Dice={best:.4f}")

    if args.model_type == 'attention':
        target_layer = model.decoders[1].conv2
        print(f"  Target layer : decoders[1].conv2  (32-ch, 32³)")
    else:
        target_layer = model.decoders[1].block[4]
        print(f"  Target layer : decoders[1].block[4]  (32-ch, 32³)")

    all_patients = sorted(glob.glob(os.path.join(DATA_ROOT, 'BraTS20_Training_*')))
    if args.n_patients is not None:
        all_patients = all_patients[:args.n_patients]
    _, sample = train_test_split(all_patients, test_size=0.09, random_state=SEED)
    print(f"  Test set : {len(sample)} pasien\n")

    for p_idx, pdir in enumerate(sample, 1):
        pid = os.path.basename(pdir)
        print(f"\n{'='*55}")
        print(f"[Patient {p_idx}/{len(sample)}] {pid}")

        image_t, gt = load_patient(pdir)
        image_t     = image_t.to(DEVICE)

        with torch.no_grad():
            pred = model(image_t).argmax(dim=1).squeeze(0).cpu().numpy()

        tumour_count = [(gt[:, :, s] > 0).sum() for s in range(gt.shape[-1])]
        top3_sl      = sorted(np.argsort(tumour_count)[::-1][:3].tolist())
        print(f"  Top-3 tumour slices: z={top3_sl}")

        heatmaps_by_class = {}

        for class_idx, class_name in CLASS_NAMES.items():
            if (gt == class_idx).sum() == 0:
                print(f"  Skipping {class_name} — no GT voxels")
                continue
            print(f"\n  Class: {class_name.replace('_', ' ')}")
            heatmaps_by_class[class_idx] = compute_all_cams(
                model, target_layer, image_t, class_idx)

        if heatmaps_by_class:
            plot_cam_combined(image_t.squeeze(0).cpu().numpy(),
                              gt, pred, heatmaps_by_class,
                              pid, top3_sl, OUTPUT_DIR,
                              arch_tag=args.model_type)

    print(f"\nAll results saved to:\n  {OUTPUT_DIR}")
