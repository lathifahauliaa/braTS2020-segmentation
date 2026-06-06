"""
BraTS 2020 - CAM Methods Comparison (Custom 3D Implementation)
Methods: GradCAM | GradCAM++ | ScoreCAM | XGradCAM | AblationCAM

Run: python cam_visualization.py
"""

import matplotlib
matplotlib.use('Agg')

import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter

# =============================================================================
# CONFIG
# =============================================================================
IS_COLAB = os.path.exists('/content')

if IS_COLAB:
    COLAB_DRIVE_ROOT = '/content/drive/MyDrive/skripsi'
    DATA_ROOT  = f'{COLAB_DRIVE_ROOT}/dataset/brats/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    CKPT_PATH  = f'{COLAB_DRIVE_ROOT}/checkpoints/best_model.pth'
    OUTPUT_DIR = f'{COLAB_DRIVE_ROOT}/cam_results'
else:
    DATA_ROOT  = r'D:\skripsi\dataset\brats\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
    CKPT_PATH  = r'D:\skripsi\checkpoints\best_model.pth'
    OUTPUT_DIR = r'D:\skripsi\cam_results'

CROP_SIZE  = (128, 128, 128)
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_PATIENTS = 80
MODALITIES = ['flair', 't1', 't1ce', 't2']
CLASS_NAMES = {1: 'Necrotic_Core', 2: 'Oedema', 3: 'Enhancing_Tumour'}

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
    image = crop_center(np.stack(vols, axis=0), CROP_SIZE)
    seg_path = glob.glob(os.path.join(patient_dir, '*seg*.nii*'))[0]
    seg = crop_center(remap_labels(load_nii(seg_path)), CROP_SIZE)
    return torch.from_numpy(image).float().unsqueeze(0), seg


# =============================================================================
# BASE CAM 3D — hooks activations and gradients from target layer
# =============================================================================
class BaseCAM3D:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach().cpu()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach().cpu()

        self.fwd_handle = self.target_layer.register_forward_hook(fwd_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(bwd_hook)

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def get_score(self, output, class_idx, pred_mask=None):
        """
        Score = softmax probability summed ONLY over predicted tumor voxels.
        This forces gradients to flow from the actual tumor region, not background.
        pred_mask: (1, H, W, D) bool tensor fixed from the original forward pass.
        """
        probs = torch.softmax(output, dim=1)  # (1, 4, H, W, D)
        if pred_mask is not None and pred_mask.float().sum() > 0:
            return (probs[:, class_idx] * pred_mask.float()).sum()
        # Fallback: full volume (used when model predicts no voxels for this class)
        return probs[:, class_idx].sum()

    @staticmethod
    def _pred_mask(output, class_idx):
        """Non-differentiable prediction mask — does not affect gradient flow."""
        return (output.detach().argmax(dim=1) == class_idx)  # (1, H, W, D)

    def upsample(self, cam_3d, target_size):
        """Upsample 3D CAM map (H', W', D') → (H, W, D)."""
        cam_t = torch.from_numpy(cam_3d).float().unsqueeze(0).unsqueeze(0)
        up    = F.interpolate(cam_t, size=target_size, mode='trilinear',
                              align_corners=False)
        return up.squeeze().numpy()

    def normalize(self, cam, sigma=2.0):
        cam = np.maximum(cam, 0)
        cam = gaussian_filter(cam, sigma=sigma)   # smooth blocky artefacts
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def compute(self, input_tensor, class_idx):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()


# =============================================================================
# 1. GRAD-CAM (3D)
# Reference: Selvaraju et al. 2017
# weight_c = mean( d(score) / d(A_c) )  via backward pass
# =============================================================================
class GradCAM3D(BaseCAM3D):
    def compute(self, input_tensor, class_idx):
        self.model.eval()
        print("      [fwd]", end=" ", flush=True)
        inp    = input_tensor.clone().requires_grad_(True)
        output = self.model(inp)
        mask   = self._pred_mask(output, class_idx)
        self.model.zero_grad()
        score  = self.get_score(output, class_idx, mask)
        print("[bwd - harap tunggu beberapa menit...]", end=" ", flush=True)
        score.backward()
        print("OK", flush=True)

        acts    = self.activations[0].numpy()
        grads   = self.gradients[0].numpy()
        weights = np.mean(grads, axis=(1, 2, 3))
        cam     = np.einsum('c,chwd->hwd', weights, acts)
        return self.normalize(self.upsample(cam, input_tensor.shape[2:]))


# =============================================================================
# 2. GRAD-CAM++ (3D)
# Reference: Chattopadhay et al. 2018
# Uses 2nd & 3rd order gradients via backward pass
# =============================================================================
class GradCAMPlusPlus3D(BaseCAM3D):
    def compute(self, input_tensor, class_idx):
        self.model.eval()
        print("      [fwd]", end=" ", flush=True)
        inp    = input_tensor.clone().requires_grad_(True)
        output = self.model(inp)
        mask   = self._pred_mask(output, class_idx)
        self.model.zero_grad()
        score  = self.get_score(output, class_idx, mask)
        print("[bwd - harap tunggu beberapa menit...]", end=" ", flush=True)
        score.backward()
        print("OK", flush=True)

        acts  = self.activations[0].numpy()
        grads = self.gradients[0].numpy()

        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_acts = np.sum(acts, axis=(1, 2, 3))
        eps      = 1e-6

        denom   = (2 * grads_power_2 +
                   sum_acts[:, None, None, None] * grads_power_3 + eps)
        aij     = grads_power_2 / denom
        aij     = np.where(grads != 0, aij, 0)
        weights = np.sum(np.maximum(grads, 0) * aij, axis=(1, 2, 3))

        cam = np.einsum('c,chwd->hwd', weights, acts)
        return self.normalize(self.upsample(cam, input_tensor.shape[2:]))


# =============================================================================
# 3. SCORE-CAM (3D)
# Reference: Wang et al. 2020
# Gradient-free: each activation channel is used as a mask on the input
# =============================================================================
class ScoreCAM3D(BaseCAM3D):
    def compute(self, input_tensor, class_idx, max_channels=32):
        self.model.eval()

        with torch.no_grad():
            output = self.model(input_tensor)

        # Fix mask from original prediction — used for ALL iterations
        mask           = self._pred_mask(output, class_idx)
        original_score = self.get_score(output, class_idx, mask).item()
        acts           = self.activations[0].numpy()   # (C, h, w, d)
        C              = acts.shape[0]
        inp_size       = input_tensor.shape[2:]

        norms  = np.sum(np.abs(acts), axis=(1, 2, 3))
        top_ch = np.argsort(norms)[::-1][:min(max_channels, C)]

        weights = np.zeros(C, dtype=np.float32)

        with torch.no_grad():
            for i, ch in enumerate(top_ch):
                if (i + 1) % 8 == 0:
                    print(f"      ScoreCAM channel {i+1}/{len(top_ch)}...", flush=True)

                act_ch   = torch.from_numpy(acts[ch]).float().unsqueeze(0).unsqueeze(0).to(input_tensor.device)
                act_up   = F.interpolate(act_ch, size=inp_size, mode='trilinear',
                                         align_corners=False).squeeze()
                a_min, a_max = act_up.min(), act_up.max()
                act_norm = ((act_up - a_min) / (a_max - a_min + 1e-8)
                            if a_max > a_min else act_up * 0)

                masked_input = input_tensor * act_norm.unsqueeze(0).unsqueeze(0)
                masked_out   = self.model(masked_input)
                # Use FIXED original mask for consistent comparison
                weights[ch]  = self.get_score(masked_out, class_idx, mask).item()

        cam = np.einsum('c,chwd->hwd', weights, acts)
        cam = self.normalize(self.upsample(cam, inp_size))
        return cam


# =============================================================================
# 4. XGRAD-CAM (3D)  — CPU-friendly: normalized prob-activation
# Reference: Fu et al. 2020
# weight_c = sum(acts_c * P_down) / (sum(acts_c) + eps)
# =============================================================================
class XGradCAM3D(BaseCAM3D):
    def compute(self, input_tensor, class_idx):
        self.model.eval()
        print("      [fwd]", end=" ", flush=True)
        inp    = input_tensor.clone().requires_grad_(True)
        output = self.model(inp)
        mask   = self._pred_mask(output, class_idx)
        self.model.zero_grad()
        score  = self.get_score(output, class_idx, mask)
        print("[bwd - harap tunggu beberapa menit...]", end=" ", flush=True)
        score.backward()
        print("OK", flush=True)

        acts  = self.activations[0].numpy()
        grads = self.gradients[0].numpy()

        sum_acts = np.sum(acts, axis=(1, 2, 3))
        weights  = (grads * acts / (sum_acts[:, None, None, None] + 1e-7)).sum(axis=(1, 2, 3))

        cam = np.einsum('c,chwd->hwd', weights, acts)
        return self.normalize(self.upsample(cam, input_tensor.shape[2:]))


# =============================================================================
# 5. ABLATION-CAM (3D)
# Reference: Desai & Ramaswamy, WACV 2020
# Gradient-free: ablate each channel and measure score drop
# =============================================================================
class AblationCAM3D(BaseCAM3D):
    def compute(self, input_tensor, class_idx, max_channels=32):
        self.model.eval()

        with torch.no_grad():
            output = self.model(input_tensor)

        # Fix mask from original prediction — used for ALL ablation iterations
        mask           = self._pred_mask(output, class_idx)
        original_score = self.get_score(output, class_idx, mask).item()

        acts   = self.activations[0]   # (C, h, w, d) tensor
        C      = acts.shape[0]
        norms  = acts.abs().sum(dim=(1, 2, 3)).numpy()
        top_ch = np.argsort(norms)[::-1][:min(max_channels, C)]

        weights = np.full(C, original_score, dtype=np.float32)

        with torch.no_grad():
            for i, ch in enumerate(top_ch):
                if (i + 1) % 8 == 0:
                    print(f"      AblationCAM channel {i+1}/{len(top_ch)}...", flush=True)

                def make_hook(channel):
                    def hook(module, inp, out):
                        out = out.clone(); out[:, channel] = 0; return out
                    return hook

                handle        = self.target_layer.register_forward_hook(make_hook(ch))
                ablated_out   = self.model(input_tensor)
                # Use FIXED original mask for consistent comparison
                ablated_score = self.get_score(ablated_out, class_idx, mask).item()
                handle.remove()
                weights[ch]   = ablated_score

        # Score drop = original - ablated (higher drop → channel more important)
        weights = original_score - weights
        cam     = np.einsum('c,chwd->hwd', weights, acts.numpy())
        cam     = self.normalize(self.upsample(cam, input_tensor.shape[2:]))
        return cam


# =============================================================================
# VISUALISATION — all 5 CAMs in one figure
# =============================================================================
def _colored_cam_slice(heatmaps_by_class, cam_name, pred_sl, sl):
    """
    Combine all 3 class CAMs into one RGBA overlay for a single axial slice.
    Color = viridis(class/3)  →  same palette as GT/Pred.
    Alpha = normalized CAM activation of that class at each voxel.
    """
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

        color            = np.array(viridis_cmap(class_idx / 3.0))  # RGBA
        rgba[mask, :3]   = color[:3]
        rgba[mask, 3]    = np.clip(cam_sl[mask] * 0.9, 0.15, 0.9)  # visible even at low cam

    return rgba


def plot_cam_combined(image_np, gt, pred, heatmaps_by_class, pid, slices):
    """
    ONE figure per patient combining all 3 tumour classes.
    Rows  = slices (top-3 tumour slices).
    Cols  = FLAIR | GT | Pred | GradCAM | GradCAM++ | ScoreCAM | XGradCAM | AblationCAM
    CAM overlay colour = viridis class colour (same as GT / Pred).
    """
    cam_names = ['GradCAM', 'GradCAM++', 'ScoreCAM', 'XGradCAM', 'AblationCAM']
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

        # FLAIR
        axes[row, 0].imshow(flair, cmap='gray', origin='lower')
        axes[row, 0].axis('off')

        # Ground Truth — viridis (same as visualize.py)
        im_gt = axes[row, 1].imshow(gt_sl, cmap='viridis',
                                    origin='lower', vmin=0, vmax=3)
        axes[row, 1].axis('off')

        # Prediction — viridis
        axes[row, 2].imshow(pr_sl, cmap='viridis',
                            origin='lower', vmin=0, vmax=3)
        axes[row, 2].axis('off')

        # Pre-compute per-class binary masks and viridis colors for contours
        viridis_cmap   = plt.cm.get_cmap('viridis')
        class_contours = {
            c: {
                'gt_mask' : (gt_sl == c).astype(np.float32),
                'pr_mask' : (pr_sl == c).astype(np.float32),
                'color'   : viridis_cmap(c / 3.0)[:3],
            }
            for c in [1, 2, 3]
        }

        # Add contours to GT column
        for c, info in class_contours.items():
            if info['gt_mask'].max() > 0:
                axes[row, 1].contour(info['gt_mask'], levels=[0.5],
                                     colors=[info['color']], linewidths=1.8,
                                     origin='lower')

        # Add contours to Pred column
        for c, info in class_contours.items():
            if info['pr_mask'].max() > 0:
                axes[row, 2].contour(info['pr_mask'], levels=[0.5],
                                     colors=[info['color']], linewidths=1.8,
                                     origin='lower')

        # 5 CAM columns — combined colour overlay on FLAIR + contours
        for col, name in enumerate(cam_names, start=3):
            axes[row, col].imshow(flair, cmap='gray', origin='lower')
            rgba = _colored_cam_slice(heatmaps_by_class, name, pr_sl, sl)
            axes[row, col].imshow(rgba, origin='lower')
            # Draw class contours so boundaries are crisp
            for c, info in class_contours.items():
                if info['pr_mask'].max() > 0:
                    axes[row, col].contour(info['pr_mask'], levels=[0.5],
                                           colors=[info['color']], linewidths=1.8,
                                           origin='lower')
            axes[row, col].axis('off')

    # Shared viridis colorbar (same as GT/Pred)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=3))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cb = fig.colorbar(sm, cax=cbar_ax, ticks=[0, 1, 2, 3])
    cb.set_ticklabels(['Background', 'Necrotic', 'Oedema', 'Enhancing'])
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(f'Patient: {pid}  —  CAM Comparison (All Classes)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.subplots_adjust(left=0.05, right=0.91, hspace=0.05, wspace=0.04)

    fname = os.path.join(OUTPUT_DIR, f'{pid}_CAM.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved -> {fname}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_patients', type=int, default=N_PATIENTS,
                        help='Jumlah pasien yang diproses (default: 80)')
    args = parser.parse_args()

    print("=" * 65)
    print("  BraTS 2020 — Custom 3D CAM Comparison")
    print("  GradCAM | GradCAM++ | ScoreCAM | XGradCAM | AblationCAM")
    print("=" * 65)

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"No checkpoint at:\n  {CKPT_PATH}")

    print("\nLoading model...")
    model = AttentionUNet3D(in_channels=4, out_channels=4).to(DEVICE)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    best = ckpt.get('best_dice', ckpt.get('best_mean_dice', 0))
    print(f"  Epoch={ckpt.get('epoch','?')}   Best Dice={best:.4f}")

    target_layer = model.decoders[1].conv2
    print(f"  Target layer: decoders[1].conv2  (32x32x32)")

    patients = sorted(glob.glob(os.path.join(DATA_ROOT, 'BraTS20_Training_*')))
    sample   = patients[:args.n_patients]
    print(f"  Memproses {len(sample)} pasien")

    for p_idx, pdir in enumerate(sample, 1):
        pid = os.path.basename(pdir)
        print(f"\n{'='*55}")
        print(f"[Patient {p_idx}/{len(sample)}] {pid}")

        image_t, gt = load_patient(pdir)
        image_t     = image_t.to(DEVICE)

        with torch.no_grad():
            pred = model(image_t).argmax(dim=1).squeeze(0).cpu().numpy()

        # Top-3 slices with most tumour voxels
        tumour_count = [(gt[:, :, s] > 0).sum() for s in range(gt.shape[-1])]
        top3_sl      = sorted(np.argsort(tumour_count)[::-1][:3].tolist())
        print(f"  Top-3 tumour slices: z={top3_sl}")

        # heatmaps_by_class[class_idx][method_name] = 3-D numpy array
        heatmaps_by_class = {}

        for class_idx, class_name in CLASS_NAMES.items():
            if (gt == class_idx).sum() == 0:
                print(f"  Skipping {class_name} — no GT voxels")
                continue

            print(f"\n  Class: {class_name.replace('_', ' ')}")
            hm = {}

            print("    [1/5] GradCAM...")
            with GradCAM3D(model, target_layer) as cam:
                hm['GradCAM'] = cam.compute(image_t, class_idx)

            print("    [2/5] GradCAM++...")
            with GradCAMPlusPlus3D(model, target_layer) as cam:
                hm['GradCAM++'] = cam.compute(image_t, class_idx)

            print("    [3/5] ScoreCAM...")
            with ScoreCAM3D(model, target_layer) as cam:
                hm['ScoreCAM'] = cam.compute(image_t, class_idx, max_channels=8)

            print("    [4/5] XGradCAM...")
            with XGradCAM3D(model, target_layer) as cam:
                hm['XGradCAM'] = cam.compute(image_t, class_idx)

            print("    [5/5] AblationCAM...")
            with AblationCAM3D(model, target_layer) as cam:
                hm['AblationCAM'] = cam.compute(image_t, class_idx, max_channels=8)

            heatmaps_by_class[class_idx] = hm

        # ONE combined figure per patient (all classes + all methods)
        if heatmaps_by_class:
            plot_cam_combined(image_t.squeeze(0).cpu().numpy(),
                              gt, pred, heatmaps_by_class,
                              pid, top3_sl)

    print(f"\nAll results saved to:\n  {OUTPUT_DIR}")
