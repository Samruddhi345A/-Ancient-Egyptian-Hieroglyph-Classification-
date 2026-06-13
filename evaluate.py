"""
evaluate.py  –  Full evaluation: accuracy, precision, recall, F1,
                per-class accuracy, confusion matrix, confusable pair analysis.
"""
import os
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_recall_fscore_support)

from utils import plot_confusion_matrix, save_results


@torch.no_grad()
def get_predictions(model: nn.Module,
                    loader: DataLoader,
                    device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and collect all predictions and ground-truth labels."""
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds: np.ndarray,
                    labels: np.ndarray,
                    class_names: List[str]) -> Dict:
    """Compute full suite of classification metrics."""
    acc = accuracy_score(labels, preds)
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    # Per-class
    prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": float(prec_pc[i]),
            "recall": float(rec_pc[i]),
            "f1": float(f1_pc[i]),
        }
        for i in range(len(class_names))
        if i < len(prec_pc)
    }

    return {
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "per_class": per_class,
    }


def find_confusable_pairs(cm: np.ndarray,
                           class_names: List[str],
                           top_k: int = 20) -> List[Dict]:
    """
    Identify the most frequently confused class pairs.
    Section 4.5 of the study — confusable glyph pair analysis.
    """
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true":      class_names[i] if i < len(class_names) else str(i),
                    "predicted": class_names[j] if j < len(class_names) else str(j),
                    "count":     int(cm[i, j]),
                    "true_total": int(cm[i].sum()),
                    "error_rate": float(cm[i, j] / max(cm[i].sum(), 1)),
                })
    # Sort by count descending
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_k]


def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   class_names: List[str],
                   model_name: str,
                   cfg,
                   device: torch.device) -> Dict:
    """
    Full evaluation pipeline for one model.
    Saves: metrics JSON, confusion matrix PNG, confusable pairs JSON.
    Returns metrics dict.
    """
    print(f"\n[Evaluate] {model_name}")
    preds, labels = get_predictions(model, test_loader, device)
    metrics = compute_metrics(preds, labels, class_names)
    cm = confusion_matrix(labels, preds)

    # Print summary
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['macro_precision']:.4f}")
    print(f"  Recall:    {metrics['macro_recall']:.4f}")
    print(f"  F1-Score:  {metrics['macro_f1']:.4f}")

    # Confusable pairs
    pairs = find_confusable_pairs(cm, class_names)
    print(f"\n  Top-5 confusable pairs:")
    for p in pairs[:5]:
        print(f"    {p['true']} → {p['predicted']}: "
              f"{p['count']} errors ({p['error_rate']:.1%})")

    # Save results
    out = {**metrics, "confusable_pairs": pairs}
    save_results(out,
                 os.path.join(cfg.results_dir, f"{model_name}_metrics.json"))

    # Plot confusion matrix
    plot_confusion_matrix(
        cm, class_names, model_name,
        os.path.join(cfg.plots_dir, f"{model_name}_confusion.png")
    )
    return metrics


def evaluate_by_gardiner_category(metrics: Dict,
                                   class_names: List[str]) -> Dict[str, float]:
    """
    Group per-class F1 by Gardiner category (first letter of class name).
    Mirrors Table 2 of Fuentes-Ferrer et al. (2025).
    """
    categories = {}
    for name, m in metrics.get("per_class", {}).items():
        cat = name[0].upper() if name else "?"
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(m["f1"])
    return {cat: float(np.mean(f1s)) for cat, f1s in categories.items()}
