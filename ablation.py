"""
ablation.py  –  Data Scarcity Threshold Analysis (Section 4.2 of the study).

For each model × data fraction:
  - Train from scratch on that fraction of training data
  - Evaluate on the FULL validation set (fixed)
  - Record accuracy
  - Identify the data-efficiency threshold:
      the fraction at which each architecture reaches 80% of its full-data accuracy

This directly answers the theoretical prediction:
  ViTs need more data than CNNs; RNNs occupy an intermediate position.
"""
import os
from typing import Dict

import torch
import torch.nn as nn

from dataset import build_ablation_loader, build_dataloaders
from models import get_model
from train import train_model, evaluate_epoch
from utils import plot_ablation_results, save_results


def run_ablation(cfg, device: torch.device) -> Dict:
    """
    Run the full data scarcity experiment.

    For speed, we run ablation only on a subset of models (configurable via
    cfg.ablation_model). To run all models set cfg.ablation_model = "all".

    Returns:
        ablation_results[model_name][fraction] = val_accuracy
    """
    # Determine which models to ablate
    if cfg.ablation_model == "all":
        model_names = ["glyphnet", "resnet50", "cnn_lstm", "vit_b16", "deit_small"]
    else:
        model_names = [cfg.ablation_model]

    # Get num_classes from a temporary full dataloader
    _, val_loader, _, num_classes, class_names = build_dataloaders(cfg)

    ablation_results: Dict[str, Dict[float, float]] = {m: {} for m in model_names}
    threshold_results: Dict[str, float] = {}

    print("\n" + "="*60)
    print("  DATA SCARCITY THRESHOLD ANALYSIS")
    print("="*60)

    for model_name in model_names:
        print(f"\n  Model: {model_name}")
        full_data_acc = None

        for fraction in sorted(cfg.ablation_fractions):
            print(f"\n    Fraction: {fraction*100:.0f}% of training data")

            # Build loaders for this fraction
            train_loader, val_loader_abl = build_ablation_loader(cfg, fraction)

            # Fresh model (no checkpoint from main training)
            model = get_model(model_name, num_classes, cfg).to(device)

            # Shorter training for ablation
            _, best_val_acc, _ = train_model(
                model, train_loader, val_loader_abl,
                cfg, f"ablation_{model_name}_{int(fraction*100)}pct",
                device, max_epochs=cfg.ablation_epochs
            )

            ablation_results[model_name][fraction] = best_val_acc

            # Record full-data accuracy as the baseline
            if fraction == 1.0 or (
                fraction == max(cfg.ablation_fractions)
            ):
                full_data_acc = best_val_acc

            print(f"    → Val accuracy at {fraction*100:.0f}%: {best_val_acc:.4f}")

        # Find 80% threshold
        if full_data_acc and full_data_acc > 0:
            target = 0.80 * full_data_acc
            fracs_sorted = sorted(cfg.ablation_fractions)
            accs = [ablation_results[model_name][f] for f in fracs_sorted]
            thr = None
            for f, a in zip(fracs_sorted, accs):
                if a >= target:
                    thr = f
                    break
            threshold_results[model_name] = thr
            print(f"\n    80%-of-full-data threshold for {model_name}: "
                  f"{thr*100 if thr else 'not reached':.0f}%")

    # Print comparison table
    _print_ablation_table(ablation_results, threshold_results, cfg.ablation_fractions)

    # Save results
    serialisable = {
        m: {str(f): v for f, v in d.items()}
        for m, d in ablation_results.items()
    }
    save_results({"ablation": serialisable, "thresholds": threshold_results},
                 os.path.join(cfg.results_dir, "ablation_results.json"))

    # Plot
    plot_ablation_results(
        ablation_results,
        os.path.join(cfg.plots_dir, "ablation_data_scarcity.png")
    )

    return ablation_results, threshold_results


def _print_ablation_table(ablation_results: Dict,
                           threshold_results: Dict,
                           fractions):
    """Print a formatted summary table."""
    models = list(ablation_results.keys())
    fracs = sorted(fractions)

    print("\n" + "="*70)
    print("DATA SCARCITY RESULTS — Validation Accuracy")
    print("="*70)
    header = f"{'Model':22s}"
    for f in fracs:
        header += f"  {f*100:5.0f}%"
    header += "  80%-Threshold"
    print(header)
    print("-"*70)
    for m in models:
        row = f"{m:22s}"
        for f in fracs:
            val = ablation_results[m].get(f, float("nan"))
            row += f"  {val:6.4f}"
        thr = threshold_results.get(m)
        row += f"  {thr*100 if thr else 'N/A':>12}"
        print(row)
    print("="*70)
