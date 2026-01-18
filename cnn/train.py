import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from cnn.dataset import CraterDataset
from cnn.model import CraterRefiner


# =========================
# CONFIG
# =========================
GT_CSV = "../nasa-craters-data/train-gt.csv"
CV_CSV = "../crater_baseline/output/solution.csv"
IMG_ROOT = "../nasa-craters-data/train"

BATCH_SIZE = 64
EPOCHS = 12
LR = 1e-4
NUM_WORKERS = 4
CROP_SIZE = 96

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "crater_refiner.pth"


# =========================
# DATASET & LOADER
# =========================
dataset = CraterDataset(
    gt_csv=GT_CSV,
    cv_csv=CV_CSV,
    img_root=IMG_ROOT,
    size=CROP_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"[INFO] Dataset size: {len(dataset)} samples")


# =========================
# MODEL
# =========================
model = CraterRefiner().to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.SmoothL1Loss()  # Huber loss


# =========================
# TRAINING LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for imgs, targets in pbar:
        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader)
    print(f"[Epoch {epoch+1}] avg loss: {avg_loss:.6f}")

    # Save checkpoint each epoch
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"[INFO] Saved model to {SAVE_PATH}")

print("[DONE] Training complete")
