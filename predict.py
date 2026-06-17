"""
BraTS 2020 - Prediction & Visualisation
Run: python predict.py
Shows: FLAIR | T1-CE | Prediction | Ground Truth
"""

import os
import glob
import random
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split

# =============================================================================
# CONFIG
# =============================================================================
IS_KAGGLE = os.path.exists('/kaggle/working')
IS_COLAB  = os.path.exists('/content') and not IS_KAGGLE

if IS_KAGGLE:
    DATA_ROOT  = '/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    CKPT_PATH  = '/kaggle/working/checkpoints/best_model.pth'
    OUTPUT_DIR = '/kaggle/working/predictions'
elif IS_COLAB:
    COLAB_DRIVE_ROOT = '/content/drive/MyDrive/skripsi'
    DATA_ROOT  = f'{COLAB_DRIVE_ROOT}/dataset/brats/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    CKPT_PATH  = f'{COLAB_DRIVE_ROOT}/checkpoints/best_model.pth'
    OUTPUT_DIR = f'{COLAB_DRIVE_ROOT}/predictions'
else:
    DATA_ROOT  = r'D:\skripsi\dataset\brats\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
    CKPT_PATH  = r'D:\skripsi\checkpoints\best_model.pth'
    OUTPUT_DIR = r'D:\skripsi\predictions'

CROP_SIZE  = (128, 128, 128)
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_PATIENTS = 64                    # visualise all 64 training patients
N_SLICES   = 5                     # how many slices per patient

LABEL_COLOURS = np.array([
    [0.267, 0.005, 0.329, 1.0],  # background  - dark purple  (viridis 0.0)
    [0.192, 0.408, 0.557, 1.0],  # necrosis    - dark blue    (viridis 0.33)
    [0.208, 0.718, 0.475, 1.0],  # oedema      - teal green   (viridis 0.67)
    [0.992, 0.906, 0.145, 1.0],  # enhancing   - yellow       (viridis 1.0)
], dtype=np.float32)

LABEL_NAMES = ['Background', 'Necrotic Core', 'Oedema', 'Enhancing Tumour']
MODALITIES  = ['flair', 't1', 't1ce', 't2']


# =============================================================================
# MODEL (copy of model.py — standalone)
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
            nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm3d(out_ch))
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
# DATA LOADING
# =============================================================================
def load_nii(path):
    img = nib.load(path)
    return np.array(img.dataobj).astype(np.float32)

def zscore(vol):
    mask = vol > 0
    if mask.sum() > 0:
        vol = np.where(mask, (vol - vol[mask].mean()) / (vol[mask].std() + 1e-8), 0.0)
    return vol

def crop_center(arr, crop=(128,128,128)):
    h, w, d = arr.shape[-3], arr.shape[-2], arr.shape[-1]
    sh = (h - crop[0]) // 2
    sw = (w - crop[1]) // 2
    sd = (d - crop[2]) // 2
    if arr.ndim == 4:
        return arr[:, sh:sh+crop[0], sw:sw+crop[1], sd:sd+crop[2]]
    return arr[sh:sh+crop[0], sw:sw+crop[1], sd:sd+crop[2]]

def remap_labels(seg):
    out = np.zeros_like(seg, dtype=np.int64)
    out[seg == 1] = 1
    out[seg == 2] = 2
    out[seg == 4] = 3
    return out

def load_patient(patient_dir):
    vols = []
    for mod in MODALITIES:
        path = glob.glob(os.path.join(patient_dir, f'*{mod}*.nii*'))[0]
        vols.append(zscore(load_nii(path)))
    image = crop_center(np.stack(vols, axis=0), CROP_SIZE)

    seg_path = glob.glob(os.path.join(patient_dir, '*seg*.nii*'))[0]
    seg = crop_center(remap_labels(load_nii(seg_path)), CROP_SIZE)

    image_t = torch.from_numpy(image).float().unsqueeze(0)  # (1,4,H,W,D)
    return image_t, seg   # tensor, numpy


# =============================================================================
# METRICS
# =============================================================================
def compute_dice(pred_labels, gt, smooth=1e-5):
    scores = {}
    for name, labels in [('WT',[1,2,3]), ('TC',[1,3]), ('ET',[3])]:
        p = np.isin(pred_labels, labels).astype(float).ravel()
        t = np.isin(gt,          labels).astype(float).ravel()
        inter = (p * t).sum()
        scores[name] = round((2*inter + smooth) / (p.sum() + t.sum() + smooth), 4)
    return scores


# =============================================================================
# VISUALISATION
# =============================================================================
def show_patient(image_np, pred, gt, patient_id, n_slices=5):
    """
    Shows n_slices axial slices with 4 columns:
    FLAIR | T1-CE | Prediction | Ground Truth
    """
    D      = image_np.shape[-1]
    gap    = D // (n_slices + 1)
    slices = [gap * (i + 1) for i in range(n_slices)]

    fig, axes = plt.subplots(n_slices, 4, figsize=(18, 4 * n_slices))
    fig.suptitle(f'Patient: {patient_id}', fontsize=14, fontweight='bold')

    # Column headers
    for col, title in enumerate(['FLAIR (Input)', 'T1-CE (Input)',
                                  'Prediction', 'Ground Truth']):
        axes[0, col].set_title(title, fontsize=11, fontweight='bold', pad=10)

    for row, sl in enumerate(slices):
        flair = image_np[0, :, :, sl]   # FLAIR modality
        t1ce  = image_np[2, :, :, sl]   # T1-CE modality

        # Column 1 — FLAIR
        axes[row, 0].imshow(flair, cmap='gray', origin='lower')
        axes[row, 0].set_ylabel(f'Slice z={sl}', fontsize=9)
        axes[row, 0].axis('off')

        # Column 2 — T1-CE
        axes[row, 1].imshow(t1ce, cmap='gray', origin='lower')
        axes[row, 1].axis('off')

        # Column 3 — Prediction (solid colour map, no MRI background)
        axes[row, 2].imshow(LABEL_COLOURS[pred[:, :, sl]], origin='lower')
        axes[row, 2].axis('off')

        # Column 4 — Ground Truth (solid colour map, no MRI background)
        axes[row, 3].imshow(LABEL_COLOURS[gt[:, :, sl]], origin='lower')
        axes[row, 3].axis('off')

    # Legend
    patches = [mpatches.Patch(color=LABEL_COLOURS[i, :3], label=LABEL_NAMES[i])
               for i in range(1, 4)]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f'{patient_id}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved → {save_path}")

    plt.show()   # show in VS Code window
    plt.close(fig)


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Results', fontsize=14, fontweight='bold')

    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'],   label='Val Loss',   color='orange')
    axes[0].set_title('Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    colors = {'WT': 'green', 'TC': 'red', 'ET': 'blue'}
    for r in ['WT', 'TC', 'ET']:
        axes[1].plot(history[r], label=f'Dice {r}', color=colors[r])
    axes[1].set_title('BraTS Dice Score per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Training curves saved → {save_path}")
    plt.show()
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  BraTS 2020 — Prediction & Visualisation")
    print("=" * 60)

    # 1. Load model
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found:\n  {CKPT_PATH}\nRun train.py first!")

    print(f"\nLoading model from:\n  {CKPT_PATH}")
    model = AttentionUNet3D(in_channels=4, out_channels=4).to(DEVICE)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    best_dice = ckpt.get('best_dice', ckpt.get('best_mean_dice', 0))
    epoch     = ckpt.get('epoch', '?')
    print(f"  Loaded epoch={epoch}  best mean Dice={best_dice:.4f}")

    # 2. Plot training curves
    if 'history' in ckpt:
        print("\nPlotting training curves...")
        plot_training_curves(ckpt['history'])

    # 3. Predict and visualise — replicate exact same split as train.py
    all_patients = sorted(glob.glob(os.path.join(DATA_ROOT, 'BraTS20_Training_*')))[:80]
    if not all_patients:
        raise FileNotFoundError(f"No patients found in:\n  {DATA_ROOT}")

    train_dirs, _ = train_test_split(all_patients, test_size=0.2, random_state=42)
    sample = sorted(train_dirs)[:N_PATIENTS]
    print(f"\nVisualising {len(sample)} patients...\n")

    for pdir in sample:
        pid = os.path.basename(pdir)
        print(f"Processing: {pid}")

        # Load data
        image_t, gt = load_patient(pdir)

        # Predict
        with torch.no_grad():
            logits = model(image_t.to(DEVICE))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Dice scores
        scores = compute_dice(pred, gt)
        print(f"  WT={scores['WT']:.4f}  TC={scores['TC']:.4f}  ET={scores['ET']:.4f}")

        # Visualise
        show_patient(image_t.squeeze(0).numpy(), pred, gt, pid, n_slices=N_SLICES)

    print("\nDone! Check your results in:")
    print(f"  {OUTPUT_DIR}")
