"""
BraTS 2020 - Visualization
Style: T1ce | FLAIR | T2 | Ground Truth | Prediction  (viridis colormap)

Run: python visualize.py
"""

import matplotlib
matplotlib.use('Agg')   # save to file only, no window popup
import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# CONFIG
# =============================================================================
IS_KAGGLE = os.path.exists('/kaggle/working')
IS_COLAB  = os.path.exists('/content') and not IS_KAGGLE

if IS_KAGGLE:
    DATA_ROOT  = '/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    CKPT_PATH  = '/kaggle/working/checkpoints/best_model.pth'
    OUTPUT_DIR = '/kaggle/working/visualizations'
elif IS_COLAB:
    COLAB_DRIVE_ROOT = '/content/drive/MyDrive/skripsi'
    DATA_ROOT  = f'{COLAB_DRIVE_ROOT}/dataset/brats/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    CKPT_PATH  = f'{COLAB_DRIVE_ROOT}/checkpoints/best_model.pth'
    OUTPUT_DIR = f'{COLAB_DRIVE_ROOT}/visualizations'
else:
    DATA_ROOT  = r'D:\skripsi\dataset\brats\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
    CKPT_PATH  = r'D:\skripsi\checkpoints\best_model.pth'
    OUTPUT_DIR = r'D:\skripsi\visualizations'

CROP_SIZE  = (128, 128, 128)
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_PATIENTS = 80    # match training patient count
MODALITIES = ['flair', 't1', 't1ce', 't2']   # index: 0=flair,1=t1,2=t1ce,3=t2

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# MODEL
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
    h, w, d   = arr.shape[-3], arr.shape[-2], arr.shape[-1]
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
    image = crop_center(np.stack(vols, axis=0), CROP_SIZE)  # (4, H, W, D)
    seg_path = glob.glob(os.path.join(patient_dir, '*seg*.nii*'))[0]
    seg = crop_center(remap_labels(load_nii(seg_path)), CROP_SIZE)  # (H, W, D)
    return torch.from_numpy(image).float().unsqueeze(0), seg        # (1,4,H,W,D), (H,W,D)


# =============================================================================
# METRICS
# =============================================================================
def compute_dice(pred, gt, smooth=1e-5):
    scores = {}
    for name, labels in [('WT', [1,2,3]), ('TC', [1,3]), ('ET', [3])]:
        p = np.isin(pred, labels).astype(float).ravel()
        t = np.isin(gt,   labels).astype(float).ravel()
        inter = (p * t).sum()
        scores[name] = (2*inter + smooth) / (p.sum() + t.sum() + smooth)
    return scores


# =============================================================================
# PLOT 1 — Training Curves
# =============================================================================
def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History', fontsize=14, fontweight='bold', y=1.02)

    axes[0].plot(history['train_loss'], label='Train Loss', color='royalblue', linewidth=2)
    axes[0].plot(history['val_loss'],   label='Val Loss',   color='tomato',    linewidth=2)
    axes[0].set_title('Loss per Epoch', fontsize=12, pad=10)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3)

    colors = {'WT': 'green', 'TC': 'tomato', 'ET': 'royalblue'}
    for r in ['WT', 'TC', 'ET']:
        axes[1].plot(history[r], label=f'Dice {r}', color=colors[r], linewidth=2)
    axes[1].set_title('BraTS Dice Score per Epoch', fontsize=12, pad=10)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Dice Score')
    axes[1].set_ylim(0, 1); axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved -> {path}")
    plt.show()
    plt.close(fig)


# =============================================================================
# PLOT 2 — T1ce | FLAIR | T2 | Ground Truth | Prediction  (viridis style)
# =============================================================================
def plot_comparison(image_np, pred, gt, pid, n_slices=5):
    """
    For each slice shows 5 columns:
    T1ce | FLAIR | T2 | Ground Truth | Prediction
    Matches the reference code style using viridis colormap.
    """
    dice = compute_dice(pred, gt)

    # Pick slices with most tumour content
    tumour_per_slice = [(gt[:, :, s] > 0).sum() for s in range(gt.shape[-1])]
    top_slices       = sorted(range(len(tumour_per_slice)),
                              key=lambda i: tumour_per_slice[i], reverse=True)[:n_slices]
    top_slices       = sorted(top_slices)

    fig, axes = plt.subplots(n_slices, 5, figsize=(20, 4 * n_slices))

    # Column headers (only on first row)
    col_titles = ['T1ce', 'FLAIR', 'T2', 'Ground Truth\n(All Classes)', 'Predicted Mask']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight='bold', pad=8)

    for row, sl in enumerate(top_slices):
        # Modality indices: flair=0, t1=1, t1ce=2, t2=3
        t1ce  = image_np[2, :, :, sl]   # T1ce
        flair = image_np[0, :, :, sl]   # FLAIR
        t2    = image_np[3, :, :, sl]   # T2

        gt_sl = gt[:, :, sl]
        pr_sl = pred[:, :, sl]

        # Row label
        axes[row, 0].set_ylabel(f'Slice z={sl}', fontsize=9, labelpad=5)

        # Col 0 — T1ce
        axes[row, 0].imshow(t1ce,  cmap='gray', origin='lower')
        axes[row, 0].axis('off')

        # Col 1 — FLAIR
        axes[row, 1].imshow(flair, cmap='gray', origin='lower')
        axes[row, 1].axis('off')

        # Col 2 — T2
        axes[row, 2].imshow(t2,    cmap='gray', origin='lower')
        axes[row, 2].axis('off')

        # Col 3 — Ground Truth (viridis)
        im_gt = axes[row, 3].imshow(gt_sl, cmap='viridis', origin='lower', vmin=0, vmax=3)
        axes[row, 3].axis('off')

        # Col 4 — Prediction (viridis)
        im_pr = axes[row, 4].imshow(pr_sl, cmap='viridis', origin='lower', vmin=0, vmax=3)
        axes[row, 4].axis('off')

    # Colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=3))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Background', 'Necrotic', 'Oedema', 'Enhancing'])
    cbar.ax.tick_params(labelsize=9)

    # Title with dice scores
    fig.suptitle(
        f'Patient: {pid}\n'
        f'WT Dice={dice["WT"]:.3f}   TC Dice={dice["TC"]:.3f}   ET Dice={dice["ET"]:.3f}',
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.subplots_adjust(left=0.06, right=0.90, top=0.93, bottom=0.03,
                        hspace=0.08, wspace=0.05)

    path = os.path.join(OUTPUT_DIR, f'{pid}_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved -> {path}")
    plt.show()
    plt.close(fig)


# =============================================================================
# PLOT 3 — Middle slice only (like reference code, clean single row)
# =============================================================================
def plot_middle_slice(image_np, pred, gt, pid):
    """Exactly matches the reference code style — single row, middle slice."""
    dice         = compute_dice(pred, gt)
    middle_slice = image_np.shape[-1] // 2   # middle of depth axis

    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(
        f'Patient: {pid}   (Middle Slice)\n'
        f'WT={dice["WT"]:.3f}   TC={dice["TC"]:.3f}   ET={dice["ET"]:.3f}',
        fontsize=12, fontweight='bold', y=1.03
    )

    # T1ce
    ax[0].imshow(image_np[2, :, :, middle_slice], cmap='gray', origin='lower')
    ax[0].set_title('T1ce', fontsize=11, fontweight='bold', pad=8)
    ax[0].axis('off')

    # FLAIR
    ax[1].imshow(image_np[0, :, :, middle_slice], cmap='gray', origin='lower')
    ax[1].set_title('FLAIR', fontsize=11, fontweight='bold', pad=8)
    ax[1].axis('off')

    # T2
    ax[2].imshow(image_np[3, :, :, middle_slice], cmap='gray', origin='lower')
    ax[2].set_title('T2', fontsize=11, fontweight='bold', pad=8)
    ax[2].axis('off')

    # Ground Truth (viridis)
    im = ax[3].imshow(gt[:, :, middle_slice], cmap='viridis', origin='lower', vmin=0, vmax=3)
    ax[3].set_title('Ground Truth\n(All Classes)', fontsize=11, fontweight='bold', pad=8)
    ax[3].axis('off')

    # Prediction (viridis)
    ax[4].imshow(pred[:, :, middle_slice], cmap='viridis', origin='lower', vmin=0, vmax=3)
    ax[4].set_title('Predicted Mask', fontsize=11, fontweight='bold', pad=8)
    ax[4].axis('off')

    # Single colorbar
    plt.colorbar(im, ax=ax[4], fraction=0.046, pad=0.04,
                 ticks=[0, 1, 2, 3]).set_ticklabels(
                     ['Background', 'Necrotic', 'Oedema', 'Enhancing'])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'{pid}_middle_slice.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved -> {path}")
    plt.show()
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  BraTS 2020 — Visualization")
    print("=" * 60)

    # Load model
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found:\n  {CKPT_PATH}")

    print(f"\nLoading model from checkpoint...")
    model = AttentionUNet3D(in_channels=4, out_channels=4).to(DEVICE)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    best_dice = ckpt.get('best_dice', ckpt.get('best_mean_dice', 0))
    print(f"  Epoch={ckpt.get('epoch','?')}   Best Dice={best_dice:.4f}")

    # Plot training curves
    if 'history' in ckpt:
        print("\n[1] Plotting training curves...")
        plot_training_curves(ckpt['history'])

    # Load and visualize patients
    patients = sorted(glob.glob(os.path.join(DATA_ROOT, 'BraTS20_Training_*')))
    sample   = patients[:N_PATIENTS]   # same 20 patients used in training

    for idx, pdir in enumerate(sample, 1):
        pid = os.path.basename(pdir)
        print(f"\n[Patient {idx}/{len(sample)}] {pid}")

        image_t, gt = load_patient(pdir)

        with torch.no_grad():
            logits = model(image_t.to(DEVICE))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        dice = compute_dice(pred, gt)
        print(f"  WT={dice['WT']:.4f}  TC={dice['TC']:.4f}  ET={dice['ET']:.4f}")

        # Single middle slice (like reference code)
        print("  Plotting middle slice...")
        plot_middle_slice(image_t.squeeze(0).numpy(), pred, gt, pid)

        # Multi-slice comparison
        print("  Plotting multi-slice comparison...")
        plot_comparison(image_t.squeeze(0).numpy(), pred, gt, pid, n_slices=5)

    print(f"\nDone! Files saved to:\n  {OUTPUT_DIR}")
