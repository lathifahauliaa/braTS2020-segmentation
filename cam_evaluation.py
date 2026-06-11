"""
BraTS 2020 — CAM Evaluation Metrics
Metrics : DAUC (Deletion AUC) | IAUC (Insertion AUC) | IoU
Run     : python cam_evaluation.py
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
import pandas as pd

from model import AttentionUNet3D

# =============================================================================
# CONFIG
# =============================================================================
IS_COLAB = os.path.exists('/content')

if IS_COLAB:
    DRIVE_ROOT = '/content/drive/MyDrive/skripsi'
    DATA_ROOT  = f'{DRIVE_ROOT}/dataset/brats/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    CKPT_PATH  = f'{DRIVE_ROOT}/checkpoints/best_model.pth'
    OUTPUT_DIR = f'{DRIVE_ROOT}/cam_eval_results'
else:
    DATA_ROOT  = r'D:\skripsi\dataset\brats\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
    CKPT_PATH  = r'D:\skripsi\checkpoints\best_model.pth'
    OUTPUT_DIR = r'D:\skripsi\cam_eval_results'

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CROP_SIZE   = (128, 128, 128)
N_PATIENTS  = 80
SEED        = 42
N_STEPS     = 10          # steps for DAUC / IAUC curve (11 points: 0%..100%)
CAM_METHODS = ['GradCAM', 'GradCAM++', 'ScoreCAM', 'XGradCAM', 'AblationCAM']
CLASS_NAMES = {1: 'Necrotic', 2: 'Oedema', 3: 'Enhancing'}
MODALITIES  = ['flair', 't1', 't1ce', 't2']

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
# CAM CLASSES (minimal — computation only, no visualization)
# =============================================================================
class BaseCAM3D:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd(m, i, o):  self.activations = o.detach().cpu()
        def bwd(m, i, o):  self.gradients   = o[0].detach().cpu()
        self.fwd_h = self.target_layer.register_forward_hook(fwd)
        self.bwd_h = self.target_layer.register_full_backward_hook(bwd)

    def remove_hooks(self):
        self.fwd_h.remove(); self.bwd_h.remove()

    def get_score(self, output, class_idx, pred_mask=None):
        probs = torch.softmax(output, dim=1)
        if pred_mask is not None and pred_mask.float().sum() > 0:
            return (probs[:, class_idx] * pred_mask.float()).sum()
        return probs[:, class_idx].sum()

    @staticmethod
    def _pred_mask(output, class_idx):
        return (output.detach().argmax(dim=1) == class_idx)

    def upsample(self, cam_3d, target_size):
        t = torch.from_numpy(cam_3d).float().unsqueeze(0).unsqueeze(0)
        return F.interpolate(t, size=target_size, mode='trilinear',
                             align_corners=False).squeeze().numpy()

    def normalize(self, cam, sigma=2.0):
        cam = np.maximum(cam, 0)
        cam = gaussian_filter(cam, sigma=sigma)
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def __enter__(self): return self
    def __exit__(self, *a): self.remove_hooks()


class GradCAM3D(BaseCAM3D):
    def compute(self, x, class_idx):
        self.model.eval()
        inp    = x.clone().requires_grad_(True)
        output = self.model(inp)
        mask   = self._pred_mask(output, class_idx)
        self.model.zero_grad()
        self.get_score(output, class_idx, mask).backward()
        acts    = self.activations[0].numpy()
        weights = np.mean(self.gradients[0].numpy(), axis=(1, 2, 3))
        cam     = np.einsum('c,chwd->hwd', weights, acts)
        return self.normalize(self.upsample(cam, x.shape[2:]))


class GradCAMPlusPlus3D(BaseCAM3D):
    def compute(self, x, class_idx):
        self.model.eval()
        inp    = x.clone().requires_grad_(True)
        output = self.model(inp)
        mask   = self._pred_mask(output, class_idx)
        self.model.zero_grad()
        self.get_score(output, class_idx, mask).backward()
        acts  = self.activations[0].numpy()
        grads = self.gradients[0].numpy()
        g2    = grads ** 2; g3 = grads ** 3
        denom = 2*g2 + acts.sum(axis=(1,2,3))[:,None,None,None]*g3 + 1e-6
        aij   = np.where(grads != 0, g2/denom, 0)
        w     = np.sum(np.maximum(grads, 0) * aij, axis=(1,2,3))
        cam   = np.einsum('c,chwd->hwd', w, acts)
        return self.normalize(self.upsample(cam, x.shape[2:]))


class ScoreCAM3D(BaseCAM3D):
    def compute(self, x, class_idx, max_channels=32):
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        mask  = self._pred_mask(output, class_idx)
        acts  = self.activations[0].numpy()
        C     = acts.shape[0]
        norms = np.sum(np.abs(acts), axis=(1,2,3))
        top_ch = np.argsort(norms)[::-1][:min(max_channels, C)]
        weights = np.zeros(C, dtype=np.float32)
        with torch.no_grad():
            for ch in top_ch:
                act_ch  = torch.from_numpy(acts[ch]).float().unsqueeze(0).unsqueeze(0).to(x.device)
                act_up  = F.interpolate(act_ch, size=x.shape[2:], mode='trilinear',
                                        align_corners=False).squeeze()
                mn, mx  = act_up.min(), act_up.max()
                act_n   = (act_up - mn) / (mx - mn + 1e-8) if mx > mn else act_up * 0
                weights[ch] = self.get_score(self.model(x * act_n.unsqueeze(0).unsqueeze(0)),
                                             class_idx, mask).item()
        cam = np.einsum('c,chwd->hwd', weights, acts)
        return self.normalize(self.upsample(cam, x.shape[2:]))


class XGradCAM3D(BaseCAM3D):
    def compute(self, x, class_idx):
        self.model.eval()
        inp    = x.clone().requires_grad_(True)
        output = self.model(inp)
        mask   = self._pred_mask(output, class_idx)
        self.model.zero_grad()
        self.get_score(output, class_idx, mask).backward()
        acts  = self.activations[0].numpy()
        grads = self.gradients[0].numpy()
        w     = (grads * acts / (acts.sum(axis=(1,2,3))[:,None,None,None] + 1e-7)).sum(axis=(1,2,3))
        cam   = np.einsum('c,chwd->hwd', w, acts)
        return self.normalize(self.upsample(cam, x.shape[2:]))


class AblationCAM3D(BaseCAM3D):
    def compute(self, x, class_idx, max_channels=32):
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        mask     = self._pred_mask(output, class_idx)
        orig_s   = self.get_score(output, class_idx, mask).item()
        acts     = self.activations[0]
        norms    = acts.abs().sum(dim=(1,2,3)).numpy()
        top_ch   = np.argsort(norms)[::-1][:min(max_channels, acts.shape[0])]
        weights  = np.full(acts.shape[0], orig_s, dtype=np.float32)
        with torch.no_grad():
            for ch in top_ch:
                def hook(m, i, o, c=ch): out=o.clone(); out[:,c]=0; return out
                h = self.target_layer.register_forward_hook(hook)
                abl_s = self.get_score(self.model(x), class_idx, mask).item()
                h.remove()
                weights[ch] = abl_s
        cam = np.einsum('c,chwd->hwd', orig_s - weights, acts.numpy())
        return self.normalize(self.upsample(cam, x.shape[2:]))


CAM_CLASSES = {
    'GradCAM'    : GradCAM3D,
    'GradCAM++'  : GradCAMPlusPlus3D,
    'ScoreCAM'   : ScoreCAM3D,
    'XGradCAM'   : XGradCAM3D,
    'AblationCAM': AblationCAM3D,
}


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================
def _model_score(model, x, class_idx):
    """Mean softmax probability for class_idx over all voxels."""
    with torch.no_grad():
        out = model(x)
    return torch.softmax(out, dim=1)[:, class_idx].mean().item()


def compute_dauc(model, x, cam, class_idx, n_steps=N_STEPS):
    """
    Deletion AUC — delete most important voxels first, measure score curve.
    Lower DAUC = better (model relies heavily on the highlighted region).
    """
    H, W, D = cam.shape
    order   = np.argsort(cam.ravel())[::-1]
    total   = len(order)
    scores  = [_model_score(model, x, class_idx)]

    for s in range(1, n_steps + 1):
        k    = int(s / n_steps * total)
        mask = np.ones(total, dtype=np.float32)
        mask[order[:k]] = 0
        m3d  = torch.from_numpy(mask.reshape(H, W, D)).to(x.device)
        scores.append(_model_score(model, x * m3d.unsqueeze(0).unsqueeze(0), class_idx))

    return float(np.trapezoid(scores, np.linspace(0, 1, n_steps + 1)) if hasattr(np, 'trapezoid') else np.trapz(scores, np.linspace(0, 1, n_steps + 1)))


def compute_iauc(model, x, cam, class_idx, n_steps=N_STEPS):
    """
    Insertion AUC — insert most important voxels into blurred baseline.
    Scores normalized by full-image score and clipped to [0,1] (per Hardani et al. 2024).
    Higher IAUC = better.
    """
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
        k    = int(s / n_steps * total)
        mask = np.zeros(total, dtype=np.float32)
        mask[order[:k]] = 1
        m3d  = torch.from_numpy(mask.reshape(H, W, D)).to(x.device)
        x_ins = x_blur + m3d.unsqueeze(0).unsqueeze(0) * (x - x_blur)
        scores.append(min(_model_score(model, x_ins, class_idx) / f_full, 1.0))

    return float(np.trapezoid(scores, np.linspace(0, 1, n_steps + 1)) if hasattr(np, 'trapezoid') else np.trapz(scores, np.linspace(0, 1, n_steps + 1)))


def compute_iou(cam, gt_mask, threshold=0.5):
    """
    IoU between thresholded CAM and ground truth binary mask.
    Higher = better localization.
    """
    cam_bin = (cam >= threshold)
    gt_bin  = gt_mask.astype(bool)
    inter   = (cam_bin & gt_bin).sum()
    union   = (cam_bin | gt_bin).sum()
    return float(inter / (union + 1e-8))


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 65)
    print("  BraTS 2020 — CAM Evaluation: DAUC | IAUC | IoU")
    print("=" * 65)

    # Load model
    model = AttentionUNet3D(in_channels=4, out_channels=4).to(DEVICE)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Model loaded — Epoch {ckpt.get('epoch','?')}, "
          f"Best Dice {ckpt.get('best_mean_dice', 0):.4f}")

    target_layer = model.decoders[1].conv2

    # Test set (same split as train.py)
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

                kw = {'max_channels': 8} if method_name in ('ScoreCAM', 'AblationCAM') else {}
                with CAMClass(model, target_layer) as cam_obj:
                    cam = cam_obj.compute(image_t, class_idx, **kw)

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

    # Save to CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, 'cam_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # Summary table: mean per method × class
    print("\n" + "=" * 65)
    print("  SUMMARY — Mean per Method")
    print("=" * 65)
    summary = df.groupby(['Method', 'Class'])[['DAUC', 'IAUC', 'IoU']].mean().round(4)
    print(summary.to_string())

    print("\n" + "=" * 65)
    print("  SUMMARY — Mean per Method (all classes)")
    print("=" * 65)
    overall = df.groupby('Method')[['DAUC', 'IAUC', 'IoU']].mean().round(4)
    overall['DAUC_rank'] = overall['DAUC'].rank()       # lower is better
    overall['IAUC_rank'] = overall['IAUC'].rank(ascending=False)  # higher is better
    overall['IoU_rank']  = overall['IoU'].rank(ascending=False)   # higher is better
    print(overall.to_string())
