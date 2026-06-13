"""
utils.py  –  Logging, device selection, plotting helpers, checkpoint I/O.
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preference)


def make_dirs(cfg):
    for d in [cfg.output_dir, cfg.checkpoint_dir,
              cfg.results_dir, cfg.plots_dir, cfg.interpretability_dir]:
        os.makedirs(d, exist_ok=True)


def save_checkpoint(model, optimizer, epoch: int, val_acc: float,
                    path: str):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_acc": val_acc,
    }, path)


def load_checkpoint(model, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt["val_acc"], ckpt["epoch"]


def save_results(results: dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved → {path}")


def plot_training_curves(history: Dict[str, List[float]],
                          model_name: str, save_path: str):
    """Plot train/val loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", color="#2196F3")
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="#F44336")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train Acc", color="#2196F3")
    axes[1].plot(history["val_acc"],   label="Val Acc",   color="#F44336")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                           model_name: str, save_path: str,
                           max_classes: int = 30):
    """Plot confusion matrix (truncated to max_classes for readability)."""
    n = min(len(class_names), max_classes)
    cm_sub = cm[:n, :n]
    names_sub = class_names[:n]

    # Normalise rows
    cm_norm = cm_sub.astype(float) / (cm_sub.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(max(10, n // 2), max(8, n // 2)))
    sns.heatmap(cm_norm, annot=False, fmt=".2f",
                xticklabels=names_sub, yticklabels=names_sub,
                cmap="Blues", ax=ax, vmin=0, vmax=1)
    ax.set_title(f"Confusion Matrix — {model_name} (first {n} classes)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_benchmark_comparison(results: Dict[str, dict], save_path: str):
    """Bar chart comparing all models on key metrics."""
    models = list(results.keys())
    metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), 6))
    x = np.arange(len(models))
    width = 0.18

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results[m].get(metric, 0) for m in models]
        ax.bar(x + i * width, vals, width, label=metric.replace("_", " ").title(),
               color=color, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Architectures", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_degradation_results(deg_results: Dict, save_path: str):
    """Line plot: accuracy retention vs severity for each model & degradation type."""
    deg_types = list(list(deg_results.values())[0].keys())
    models = list(deg_results.keys())
    severities = [1, 2, 3]

    fig, axes = plt.subplots(1, len(deg_types),
                              figsize=(5 * len(deg_types), 5), sharey=True)
    if len(deg_types) == 1:
        axes = [axes]

    for ax, deg in zip(axes, deg_types):
        for model_name in models:
            vals = [deg_results[model_name][deg].get(s, 0) for s in severities]
            ax.plot(severities, vals, marker="o", label=model_name)
        ax.set_title(deg.replace("_", " ").title())
        ax.set_xlabel("Severity")
        ax.set_ylabel("Accuracy Retention (%)")
        ax.set_xticks(severities)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Degradation Robustness Benchmark", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ablation_results(ablation_results: Dict, save_path: str):
    """Plot accuracy vs training data fraction for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, data in ablation_results.items():
        fractions = sorted(data.keys())
        accs = [data[f] for f in fractions]
        ax.plot([f * 100 for f in fractions], accs, marker="o", label=model_name)

    ax.axhline(y=0.80, color="gray", linestyle="--", alpha=0.5, label="80% threshold")
    ax.set_xlabel("Training Data Used (%)")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Data Scarcity Threshold Analysis", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
