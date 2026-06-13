from typing import Optional, Dict, Tuple
"""
train.py  –  Training loop with cosine annealing LR, early stopping,
             label smoothing, and checkpoint saving.
"""
import os
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import save_checkpoint, plot_training_curves


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> Tuple[float, float]:
    """Single training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float]:
    """Single evaluation epoch. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                cfg,
                model_name: str,
                device: torch.device,
                max_epochs: Optional[int] = None) -> Dict:
    """
    Full training loop.
    Returns history dict with train/val loss and accuracy per epoch.
    """
    

    max_epochs = max_epochs or cfg.epochs
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=cfg.lr * 0.01
    )

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"{model_name}_best.pt")

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Device: {device} | Epochs: {max_epochs} | LR: {cfg.lr}")
    print(f"{'='*60}")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                           optimizer, device)
        va_loss, va_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:03d}/{max_epochs} | "
              f"Train {tr_acc:.4f} ({tr_loss:.4f}) | "
              f"Val {va_acc:.4f} ({va_loss:.4f}) | "
              f"{elapsed:.1f}s | LR {scheduler.get_last_lr()[0]:.2e}")

        # Checkpoint best model
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, va_acc, ckpt_path)
            print(f"    ✓ New best: {best_val_acc:.4f} — saved checkpoint")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= cfg.early_stopping_patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no improvement for {cfg.early_stopping_patience} epochs)")
            break

    # Save training curves
    plot_training_curves(
        history, model_name,
        os.path.join(cfg.plots_dir, f"{model_name}_training.png")
    )
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    return history, best_val_acc, ckpt_path
