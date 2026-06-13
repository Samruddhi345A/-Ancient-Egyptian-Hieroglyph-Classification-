"""
degradation.py  –  Systematic degradation robustness benchmark.
Section 3.1 (degradation test split) and Section 4 of the study.

Two-phase process per model × degradation type × severity:
  Phase A (baseline inference):
    - Apply degradation to the test set only
    - Compute accuracy retention vs clean baseline
    (no weight updates — models already trained)

  Phase B (degradation re-training):
    - Apply degradation to train + val + test splits
    - Re-train each model from its clean checkpoint (cfg.degradation_epochs)
    - Evaluate on the degraded test split
    - Compare against Phase A retention to see how much re-training helps
"""

import os
import copy
from typing import Dict

import torch
import torch.nn as nn

from dataset import build_degraded_loader, build_degraded_retrain_loaders
from evaluate import get_predictions, compute_metrics
from models import get_model
from train import train_model
from utils import plot_degradation_results, save_results


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point called by main.py
# ─────────────────────────────────────────────────────────────────────────────

def run_degradation_benchmark(
    models_dict: Dict[str, nn.Module],
    baseline_accuracies: Dict[str, float],
    class_names,
    cfg,
    device: torch.device,
    num_classes: int,
) -> Dict:
    """
    Run the full degradation benchmark (Phase A + Phase B).

    models_dict        : {model_name: already-trained model}
    baseline_accuracies: {model_name: clean test accuracy}
    num_classes        : needed to rebuild models for re-training

    Returns nested dict:
        results[model_name][deg_type][severity] = {
            "retention_inference":  float,   # Phase A
            "accuracy_retrained":   float,   # Phase B
            "retention_retrained":  float,   # Phase B vs clean baseline
        }
    """
    results = {m: {} for m in models_dict}

    for deg_type in cfg.degradation_types:
        print(f"\n[Degradation] Type: {deg_type}")
        for model_name in models_dict:
            results[model_name][deg_type] = {}

        for severity in cfg.degradation_severities:
            print(f"\n  ── Severity {severity} ──")

            # ── Phase A: inference only on degraded test set ──────────────────
            print("  [Phase A] Inference on degraded test images (no re-training)...")
            test_loader_deg = build_degraded_loader(cfg, deg_type, severity)

            for model_name, model in models_dict.items():
                preds, labels = get_predictions(model, test_loader_deg, device)
                metrics = compute_metrics(preds, labels, class_names)
                acc_a = metrics["accuracy"]
                baseline = baseline_accuracies.get(model_name, 1.0)
                retention_a = (acc_a / baseline * 100) if baseline > 0 else 0.0

                results[model_name][deg_type][severity] = {
                    "retention_inference": round(retention_a, 2),
                    "accuracy_retrained":  None,
                    "retention_retrained": None,
                }
                print(f"    {model_name:20s}: acc={acc_a:.4f}  "
                      f"retention={retention_a:.1f}%  [Phase A]")

            # ── Phase B: re-train on degraded data then re-test ───────────────
            print(f"  [Phase B] Re-training on degraded data "
                  f"({cfg.degradation_epochs} epochs)...")

            (train_loader_deg,
             val_loader_deg,
             test_loader_deg2) = build_degraded_retrain_loaders(
                cfg, deg_type, severity
            )

            # Swap epoch count temporarily
            original_epochs = cfg.epochs
            cfg.epochs = cfg.degradation_epochs

            for model_name, clean_model in models_dict.items():
                print(f"    Re-training {model_name}...")

                # Start from a fresh copy of the clean model weights
                model_retrain = copy.deepcopy(clean_model)

                retrain_tag = f"{model_name}_deg_{deg_type}_s{severity}"

                train_model(
                    model_retrain,
                    train_loader_deg,
                    val_loader_deg,
                    cfg,
                    retrain_tag,
                    device,
                )

                # Load the best checkpoint saved during re-training
                ckpt_path = os.path.join(
                    cfg.checkpoint_dir, f"{retrain_tag}_best.pt"
                )
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location=device)
                    model_retrain.load_state_dict(ckpt["model_state"])

                # Evaluate on degraded test set
                preds, labels = get_predictions(
                    model_retrain, test_loader_deg2, device
                )
                metrics_b = compute_metrics(preds, labels, class_names)
                acc_b = metrics_b["accuracy"]
                baseline = baseline_accuracies.get(model_name, 1.0)
                retention_b = (acc_b / baseline * 100) if baseline > 0 else 0.0

                results[model_name][deg_type][severity]["accuracy_retrained"] = (
                    round(acc_b, 4)
                )
                results[model_name][deg_type][severity]["retention_retrained"] = (
                    round(retention_b, 2)
                )

                ret_a = results[model_name][deg_type][severity]["retention_inference"]
                print(f"    {model_name:20s}: acc={acc_b:.4f}  "
                      f"retention={retention_b:.1f}%  [Phase B]  "
                      f"(Δ vs Phase A: {retention_b - ret_a:+.1f}%)")

            cfg.epochs = original_epochs  # restore

    # ── Save & report ─────────────────────────────────────────────────────────
    save_results(
        results,
        os.path.join(cfg.results_dir, "degradation_results.json")
    )
    plot_degradation_results(
        results,
        os.path.join(cfg.plots_dir, "degradation_benchmark.png")
    )
    _print_degradation_table(results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def _print_degradation_table(results: Dict):
    """Print a readable summary table to stdout."""
    models = list(results.keys())
    if not models:
        return

    deg_types  = list(results[models[0]].keys())
    severities = sorted(list(results[models[0]][deg_types[0]].keys()))

    print("\n" + "=" * 80)
    print("DEGRADATION ROBUSTNESS SUMMARY")
    print("  A = inference-only retention (%)    B = after re-training retention (%)")
    print("=" * 80)

    # Header
    header = f"{'Model':22s}"
    for d in deg_types:
        for s in severities:
            label = f"{d[:6]}S{s}"
            header += f"  {label:>9s}A  {label:>9s}B"
    print(header)
    print("-" * 80)

    for m in models:
        row = f"{m:22s}"
        for d in deg_types:
            for s in severities:
                entry = results[m][d].get(s, {})
                val_a = entry.get("retention_inference", 0) or 0
                val_b = entry.get("retention_retrained", 0) or 0
                row += f"  {val_a:11.1f}  {val_b:11.1f}"
        print(row)

    print("=" * 80)