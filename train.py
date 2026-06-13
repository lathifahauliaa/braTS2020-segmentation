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

Jika sesi Colab terputus, jalankan ulang — training otomatis dilanjut dari epoch terakhir
via resume.pth yang tersimpan di CHECKPOINT_DIR.
"""

import os
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
IS_COLAB = os.path.exists('/content')

# ── Paths (auto-selected by environment) ─────────────────────────────────────
if IS_COLAB:
    COLAB_DRIVE_ROOT = '/content/drive/MyDrive/skripsi'
    DATA_ROOT        = f'{COLAB_DRIVE_ROOT}/dataset/brats/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    CHECKPOINT_DIR   = f'{COLAB_DRIVE_ROOT}/checkpoints'
else:
    DATA_ROOT      = r'D:\skripsi\dataset\brats\archive\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
    CHECKPOINT_DIR = r'D:\skripsi\checkpoints'

RESUME_PATH = os.path.join(CHECKPOINT_DIR, 'resume.pth')

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
SEED    = 42

# ── Number of patients (set to None to use all 369) ───────────────────────────
MAX_PATIENTS = 250

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE      = 2 if IS_COLAB else 1   # Colab T4/A100 has more VRAM
LR              = 1e-3
WEIGHT_DECAY    = 1e-2
EPOCHS          = 100 if IS_COLAB else 30
CROP_SIZE       = (128, 128, 128)
NUM_CLASSES     = 4
DEEP_SUP_WEIGHT = 0.4
PATIENCE        = 15
NUM_WORKERS     = 0   # 0 untuk semua env: hindari DataLoader hang saat load file 3D besar

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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

    # Split 1: pisahkan test (9%) dari sisanya (91%)
    trainval_dirs, test_dirs = train_test_split(
        all_patients, test_size=0.09, random_state=SEED)

    # Split 2: dari 91%, pisahkan val (27/91 ≈ 29.7%) → train 64%, val 27%
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

            # Deep supervision: resize GT to each auxiliary output resolution
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

        # Progress dot every 5 steps
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
            pred = model(images)        # eval mode → single output tensor
            loss = criterion(pred, masks)

        total_loss += loss.item()

        scores = dice_per_region(pred, masks)
        for k in region_scores:
            region_scores[k].append(scores[k])

        all_iou.append(iou_per_class(pred, masks, num_classes=NUM_CLASSES))

    mean_scores = {k: float(np.mean(v)) for k, v in region_scores.items()}
    mean_iou    = float(np.mean(all_iou))
    return total_loss / len(loader), mean_scores, mean_iou


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  BraTS 2020 — Attention 3D U-Net Training")
    print("=" * 72)
    print(f"  Device      : {DEVICE}")
    print(f"  AMP enabled : {USE_AMP}")
    print(f"  Data root   : {DATA_ROOT}")

    train_loader, val_loader, test_loader, n_train, n_val, n_test = build_loaders()

    model     = AttentionUNet3D(in_channels=4, out_channels=NUM_CLASSES).to(DEVICE)
    criterion = CombinedLoss(num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY, amsgrad=True)
    # CosineAnnealingLR: LR turun monoton dari LR ke eta_min selama EPOCHS epoch.
    # Referensi: Isensee et al. (2021), nnU-Net, Nature Methods 18(2):203-211.
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = GradScaler('cuda', enabled=USE_AMP)

    print(f"  Train/Val/Test : {n_train} / {n_val} / {n_test} patients")
    print(f"  Parameters  : {count_parameters(model):,}")
    print(f"  Epochs      : {EPOCHS}  |  Patience: {PATIENCE}")
    print("=" * 72)

    best_mean_dice = 0.0
    no_improve     = 0
    start_epoch    = 1
    history        = {k: [] for k in ['train_loss', 'val_loss',
                                       'WT', 'TC', 'ET', 'mIoU']}

    # ── Resume dari checkpoint jika sesi Colab terputus ───────────────────────
    if os.path.exists(RESUME_PATH):
        print(f"\n  [Resume] Memuat checkpoint dari: {RESUME_PATH}")
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        best_mean_dice = ckpt['best_mean_dice']
        no_improve     = ckpt['no_improve']
        start_epoch    = ckpt['epoch'] + 1
        history        = ckpt['history']
        print(f"  [Resume] Lanjut dari epoch {start_epoch}  |  "
              f"Best Dice sejauh ini: {best_mean_dice:.4f}")
        print("=" * 72)

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch [{epoch:03d}/{EPOCHS}]")

        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, scaler)
        val_loss, dice_scores, miou = validate(model, val_loader, criterion)
        scheduler.step()

        mean_dice = float(np.mean(list(dice_scores.values())))
        lr_now    = scheduler.get_last_lr()[0]

        # Record history
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

        # ── Simpan resume checkpoint setiap epoch ─────────────────────────────
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_mean_dice':       best_mean_dice,
            'no_improve':           no_improve,
            'history':              history,
        }, RESUME_PATH)

        # ── Simpan best model ──────────────────────────────────────────────────
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
    print(f"Checkpoint saved at: {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}")

    # Evaluasi final di test set menggunakan best checkpoint
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

    return history


if __name__ == '__main__':
    main()
