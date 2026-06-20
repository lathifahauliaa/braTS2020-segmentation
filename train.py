"""
Training script — Attention 3D U-Net on BraTS 2020.

Scheduler : CosineAnnealingLR (Isensee et al., 2021 — nnU-Net, Nature Methods 18(2):203-211)
            LR turun monoton dari eta_max ke eta_min selama T_max epoch,
            tanpa warm restart → kompatibel penuh dengan early stopping.

Run on Google Colab:
    1. Mount Google Drive
    2. !git clone https://github.com/lathifahauliaa/braTS2020-segmentation.git /content/repo
    3. %cd /content/repo
    4. !pip install -r requirements.txt -q
    5. !python train.py

Run on Kaggle:
    1. Add dataset: BraTS2020 Dataset (Training + Validation) — awsaf49/brats20-dataset-training-validation
    2. New Notebook → Advanced → Accelerator: GPU T4 x2
    3. !git clone https://github.com/lathifahauliaa/braTS2020-segmentation.git /kaggle/working/repo
    4. %cd /kaggle/working/repo
    5. !pip install -r requirements.txt -q
    6. !python train.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

from dataset    import BraTS2020Dataset
from model      import AttentionUNet3D, count_parameters
from losses     import CombinedLoss
from metrics    import dice_per_region, iou_per_class
from transforms import get_train_transforms

# ── Environment detection ──────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists('/kaggle/working')
IS_COLAB  = os.path.exists('/content') and not IS_KAGGLE

# ── Paths (auto-selected by environment) ──────────────────────────────────────
if IS_KAGGLE:
    DATA_ROOT      = '/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    CHECKPOINT_DIR = '/kaggle/working/checkpoints'
elif IS_COLAB:
    COLAB_DRIVE_ROOT = '/content/drive/MyDrive/skripsi'
    DATA_ROOT        = f'{COLAB_DRIVE_ROOT}/dataset/brats/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    CHECKPOINT_DIR   = f'{COLAB_DRIVE_ROOT}/checkpoints'
else:
    DATA_ROOT      = r'D:\skripsi\dataset\brats\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
    CHECKPOINT_DIR = r'D:\skripsi\checkpoints'

VIZ_DIR  = os.path.join(CHECKPOINT_DIR, 'viz_attention')
LOG_PATH = os.path.join(CHECKPOINT_DIR, 'train_attention.log')

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
SEED    = 42

# ── Number of patients (set to None to use all 369) ───────────────────────────
MAX_PATIENTS = None

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE      = 2 if (IS_COLAB or IS_KAGGLE) else 1
LR              = 1e-3
WEIGHT_DECAY    = 1e-2
EPOCHS          = 100 if (IS_COLAB or IS_KAGGLE) else 30
CROP_SIZE       = (128, 128, 128)
NUM_CLASSES     = 4
DEEP_SUP_WEIGHT = 0.4
PATIENCE        = 15
NUM_WORKERS     = 4 if IS_KAGGLE else 0
N_VIZ_PATIENTS  = 3   # val patients to visualize after training

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


# ── Tee: print to stdout AND write to log file ────────────────────────────────
class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
    def isatty(self):
        return False


# ── Build data loaders ────────────────────────────────────────────────────────
def build_loaders():
    all_patients = sorted(glob.glob(
        os.path.join(DATA_ROOT, 'BraTS20_Training_*')))

    if not all_patients:
        raise FileNotFoundError(
            f"No patient folders found in:\n  {DATA_ROOT}\n"
            "Check that DATA_ROOT is correct and the dataset is unzipped.")

    if MAX_PATIENTS is not None:
        all_patients = all_patients[:MAX_PATIENTS]
        print(f"[Quick-run] Using {len(all_patients)} / "
              f"{len(sorted(glob.glob(os.path.join(DATA_ROOT, 'BraTS20_Training_*'))))} patients.")

    trainval_dirs, test_dirs = train_test_split(
        all_patients, test_size=0.09, random_state=SEED)
    train_dirs, val_dirs = train_test_split(
        trainval_dirs, test_size=0.27/0.91, random_state=SEED)

    pin = torch.cuda.is_available()

    train_ds = BraTS2020Dataset(train_dirs, transform=get_train_transforms(),
                                crop_size=CROP_SIZE)
    val_ds   = BraTS2020Dataset(val_dirs,  transform=None, crop_size=CROP_SIZE)
    test_ds  = BraTS2020Dataset(test_dirs, transform=None, crop_size=CROP_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,  batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=pin)

    return train_loader, val_loader, test_loader, len(train_ds), len(val_ds), len(test_ds)


# ── Training epoch ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0

    for step, (images, masks) in enumerate(loader, 1):
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=DEVICE.type, enabled=USE_AMP):
            main_out, deep_outs = model(images)
            loss = criterion(main_out, masks)

            for ds_out in deep_outs:
                ds_target = F.interpolate(
                    masks.unsqueeze(1).float(),
                    size=ds_out.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
                loss = loss + DEEP_SUP_WEIGHT * criterion(ds_out, ds_target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if step % 5 == 0 or step == len(loader):
            print(f"    step {step}/{len(loader)}  loss={loss.item():.4f}",
                  flush=True)

    return total_loss / len(loader)


# ── Validation epoch ──────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss    = 0.0
    region_scores = {'WT': [], 'TC': [], 'ET': []}
    all_iou       = []

    for images, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE,  non_blocking=True)

        with autocast(device_type=DEVICE.type, enabled=USE_AMP):
            pred = model(images)
            loss = criterion(pred, masks)

        total_loss += loss.item()

        scores = dice_per_region(pred, masks)
        for k in region_scores:
            region_scores[k].append(scores[k])

        all_iou.append(iou_per_class(pred, masks, num_classes=NUM_CLASSES))

    mean_scores = {k: float(np.mean(v)) for k, v in region_scores.items()}
    mean_iou    = float(np.mean(all_iou))
    return total_loss / len(loader), mean_scores, mean_iou


# ── Dice helper for visualization ─────────────────────────────────────────────
def _compute_dice(pred, gt, smooth=1e-5):
    scores = {}
    for name, labels in [('WT', [1, 2, 3]), ('TC', [1, 3]), ('ET', [3])]:
        p     = np.isin(pred, labels).astype(float).ravel()
        t     = np.isin(gt,   labels).astype(float).ravel()
        inter = (p * t).sum()
        scores[name] = (2 * inter + smooth) / (p.sum() + t.sum() + smooth)
    return scores


# ── Training history plot ─────────────────────────────────────────────────────
def _plot_history(history, viz_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History — Attention 3D U-Net', fontsize=13, fontweight='bold')

    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='royalblue', lw=2)
    axes[0].plot(epochs, history['val_loss'],   label='Val Loss',   color='tomato',    lw=2)
    axes[0].set_title('Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for r, c in zip(['WT', 'TC', 'ET'], ['green', 'tomato', 'royalblue']):
        axes[1].plot(epochs, history[r], label=f'Dice {r}', color=c, lw=2)
    axes[1].set_title('Dice Score per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(viz_dir, 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved -> {path}")
    plt.close(fig)


# ── Validation prediction visualization ───────────────────────────────────────
@torch.no_grad()
def _plot_val_predictions(model, val_loader, viz_dir, n_patients=N_VIZ_PATIENTS):
    model.eval()
    print(f"\n[Visualizing {n_patients} val patients]")

    for idx, (images, masks) in enumerate(val_loader):
        if idx >= n_patients:
            break

        logits = model(images.to(DEVICE))
        pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy()   # (H,W,D)
        gt     = masks.squeeze(0).cpu().numpy()                   # (H,W,D)
        img_np = images.squeeze(0).cpu().numpy()                  # (4,H,W,D)

        dice = _compute_dice(pred, gt)

        tumor_per_sl = [(gt[:, :, s] > 0).sum() for s in range(gt.shape[-1])]
        top_sl       = sorted(np.argsort(tumor_per_sl)[::-1][:3].tolist())

        fig, axes = plt.subplots(len(top_sl), 5,
                                 figsize=(20, 4 * len(top_sl)), squeeze=False)

        for c, t in enumerate(['T1ce', 'FLAIR', 'T2', 'Ground Truth', 'Prediction']):
            axes[0, c].set_title(t, fontsize=11, fontweight='bold', pad=8)

        for row, sl in enumerate(top_sl):
            axes[row, 0].imshow(img_np[2, :, :, sl], cmap='gray',    origin='lower')
            axes[row, 1].imshow(img_np[0, :, :, sl], cmap='gray',    origin='lower')
            axes[row, 2].imshow(img_np[3, :, :, sl], cmap='gray',    origin='lower')
            axes[row, 3].imshow(gt[:, :, sl],         cmap='viridis', origin='lower', vmin=0, vmax=3)
            axes[row, 4].imshow(pred[:, :, sl],       cmap='viridis', origin='lower', vmin=0, vmax=3)
            for ax in axes[row]:
                ax.axis('off')
            axes[row, 0].set_ylabel(f'z={sl}', fontsize=9)

        sm      = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=3))
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cb      = fig.colorbar(sm, cax=cbar_ax, ticks=[0, 1, 2, 3])
        cb.set_ticklabels(['Background', 'Necrotic', 'Oedema', 'Enhancing'])
        cb.ax.tick_params(labelsize=8)

        fig.suptitle(
            f'Val Patient {idx + 1} | '
            f'WT={dice["WT"]:.3f}  TC={dice["TC"]:.3f}  ET={dice["ET"]:.3f}',
            fontsize=12, fontweight='bold', y=1.01
        )
        plt.subplots_adjust(left=0.06, right=0.91, hspace=0.05, wspace=0.04)

        path = os.path.join(viz_dir, f'val_pred_{idx + 1:02d}.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        print(f"  Saved -> {path}")
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    env  = "Kaggle" if IS_KAGGLE else ("Colab" if IS_COLAB else "Local")
    logf = open(LOG_PATH, 'w', encoding='utf-8')
    _orig_stdout = sys.stdout
    sys.stdout   = _Tee(_orig_stdout, logf)

    try:
        print("=" * 72)
        print("  BraTS 2020 — Attention 3D U-Net Training")
        print("=" * 72)
        print(f"  Environment : {env}")
        print(f"  Device      : {DEVICE}")
        print(f"  AMP enabled : {USE_AMP}")
        print(f"  Data root   : {DATA_ROOT}")
        print(f"  Log file    : {LOG_PATH}")

        train_loader, val_loader, test_loader, n_train, n_val, n_test = build_loaders()

        model     = AttentionUNet3D(in_channels=4, out_channels=NUM_CLASSES).to(DEVICE)
        criterion = CombinedLoss(num_classes=NUM_CLASSES)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                      weight_decay=WEIGHT_DECAY, amsgrad=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        scaler    = GradScaler('cuda', enabled=USE_AMP)

        print(f"  Train/Val/Test : {n_train} / {n_val} / {n_test} patients")
        print(f"  Parameters     : {count_parameters(model):,}")
        print(f"  Epochs         : {EPOCHS}  |  Patience: {PATIENCE}")
        print("=" * 72)

        best_mean_dice = 0.0
        no_improve     = 0
        history        = {k: [] for k in ['train_loss', 'val_loss',
                                           'WT', 'TC', 'ET', 'mIoU']}

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch [{epoch:03d}/{EPOCHS}]")

            train_loss = train_one_epoch(model, train_loader, optimizer,
                                         criterion, scaler)
            val_loss, dice_scores, miou = validate(model, val_loader, criterion)
            scheduler.step()

            mean_dice = float(np.mean(list(dice_scores.values())))
            lr_now    = scheduler.get_last_lr()[0]

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['WT'].append(dice_scores['WT'])
            history['TC'].append(dice_scores['TC'])
            history['ET'].append(dice_scores['ET'])
            history['mIoU'].append(miou)

            print(f"  TrainLoss : {train_loss:.4f}  |  ValLoss : {val_loss:.4f}")
            print(f"  WT Dice   : {dice_scores['WT']:.4f}  |  "
                  f"TC Dice : {dice_scores['TC']:.4f}  |  "
                  f"ET Dice : {dice_scores['ET']:.4f}")
            print(f"  mIoU      : {miou:.4f}  |  LR : {lr_now:.2e}")

            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                no_improve     = 0
                ckpt_path      = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
                torch.save({
                    'epoch':                epoch,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mean_dice':       best_mean_dice,
                    'history':              history,
                }, ckpt_path)
                print(f"  --> Checkpoint saved  (mean Dice = {best_mean_dice:.4f})")
            else:
                no_improve += 1
                print(f"  No improvement {no_improve}/{PATIENCE}")
                if no_improve >= PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch}.")
                    break

        print(f"\nTraining complete.  Best mean Dice = {best_mean_dice:.4f}")
        print(f"Checkpoint: {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}")

        # ── Final evaluation on test set using best checkpoint ─────────────────
        print("\n" + "=" * 72)
        print("  Final Evaluation on Test Set")
        print("=" * 72)
        ckpt = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'),
                          map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        test_loss, test_dice, test_miou = validate(model, test_loader, criterion)
        test_mean_dice = float(np.mean(list(test_dice.values())))
        print(f"  TestLoss  : {test_loss:.4f}")
        print(f"  WT Dice   : {test_dice['WT']:.4f}  |  "
              f"TC Dice : {test_dice['TC']:.4f}  |  "
              f"ET Dice : {test_dice['ET']:.4f}")
        print(f"  Mean Dice : {test_mean_dice:.4f}  |  mIoU : {test_miou:.4f}")
        print("=" * 72)

        # ── Training curves ────────────────────────────────────────────────────
        print("\n[Generating training plots...]")
        _plot_history(history, VIZ_DIR)

        # ── Validation prediction visualization ────────────────────────────────
        _plot_val_predictions(model, val_loader, VIZ_DIR, n_patients=N_VIZ_PATIENTS)

        print(f"\nVisualizations saved to : {VIZ_DIR}")
        print(f"Training log saved to   : {LOG_PATH}")

        return history

    finally:
        sys.stdout = _orig_stdout
        logf.close()


if __name__ == '__main__':
    main()
