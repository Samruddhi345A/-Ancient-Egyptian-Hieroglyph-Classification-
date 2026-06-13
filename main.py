"""
main.py  –  Master orchestration script.
Run everything from here.

Pipeline order (--mode all):
  Step 1: Train all models on clean data  (cfg.epochs)
  Step 2: Test all models on clean data   → save baselines
  Step 3: Degradation benchmark
          Phase A – inference only on degraded test images (no re-training)
          Phase B – re-train each model on degraded data   (cfg.degradation_epochs)
                    then test on degraded test images
  Step 4: Interpretability maps
  Step 5: Ablation (data scarcity)        (cfg.ablation_epochs)

Usage:
  python main.py --mode train_all       # train & evaluate all 6 models
  python main.py --mode degradation     # degradation benchmark (both phases)
  python main.py --mode interpret       # interpretability maps
  python main.py --mode ablation        # data scarcity experiment
  python main.py --mode all             # run everything sequentially
  python main.py --mode single --model resnet50   # train one specific model
"""

import os
import argparse
import json
import time

import torch
import torch.nn as nn

from config import get_config
from dataset import build_dataloaders
from models import get_model, MODEL_FAMILIES
from train import train_model
from evaluate import evaluate_model
from degradation import run_degradation_benchmark
from interpretability import run_interpretability
from ablation import run_ablation
from utils import get_device, make_dirs, plot_benchmark_comparison, save_results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_active_models(cfg) -> list:
    """Return list of model names that are enabled in config."""
    mapping = {
        "glyphnet":        cfg.run_glyphnet,
        "resnet50":        cfg.run_resnet50,
        "efficientnet_b3": cfg.run_efficientnet_b3,
        "cnn_lstm":        cfg.run_cnn_lstm,
        "vit_b16":         cfg.run_vit_b16,
        "deit_small":      cfg.run_deit_small,
    }
    return [name for name, enabled in mapping.items() if enabled]


def load_trained_model(model_name: str, num_classes: int,
                        cfg, device: torch.device) -> nn.Module:
    """Load a model from its best checkpoint."""
    model = get_model(model_name, num_classes, cfg).to(device)
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"{model_name}_best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded checkpoint: {ckpt_path} "
              f"(val_acc={ckpt.get('val_acc', 0):.4f})")
    else:
        print(f"  [WARN] No checkpoint found for {model_name} at {ckpt_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Mode handlers
# ─────────────────────────────────────────────────────────────────────────────

def mode_train_all(cfg, device):
    """
    Step 1 + 2: Train all enabled models on clean data, then test them.
    Returns all_metrics, baseline_accuracies, num_classes, class_names.
    """
    print("\n" + "╔" + "═"*58 + "╗")
    print("║  STEP 1+2: TRAIN & TEST ALL MODELS (CLEAN DATA)" + " "*9 + "║")
    print(f"║  Epochs: {cfg.epochs:<49}║")
    print("╚" + "═"*58 + "╝")

    train_loader, val_loader, test_loader, num_classes, class_names = \
        build_dataloaders(cfg)

    model_names  = get_active_models(cfg)
    all_metrics  = {}
    baseline_accuracies = {}

    for model_name in model_names:
        print(f"\n{'─'*60}")
        print(f"  MODEL: {model_name.upper()}")
        family = _get_family(model_name)
        print(f"  Family: {family}")
        print(f"{'─'*60}")

        # Build model
        model  = get_model(model_name, num_classes, cfg)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        # ── Train on clean data (cfg.epochs) ──────────────────────────────────
        history, best_val_acc, ckpt_path = train_model(
            model, train_loader, val_loader, cfg, model_name, device
        )

        # ── Test on clean data ────────────────────────────────────────────────
        model   = load_trained_model(model_name, num_classes, cfg, device)
        metrics = evaluate_model(
            model, test_loader, class_names, model_name, cfg, device
        )
        all_metrics[model_name]         = metrics
        baseline_accuracies[model_name] = metrics["accuracy"]

    # ── Cross-model comparison ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("CLEAN BASELINE RESULTS")
    print("="*70)
    _print_results_table(all_metrics)

    plot_benchmark_comparison(
        all_metrics,
        os.path.join(cfg.plots_dir, "benchmark_comparison.png")
    )
    _print_family_summary(all_metrics)
    save_results(all_metrics,
                 os.path.join(cfg.results_dir, "all_models_results.json"))

    return all_metrics, baseline_accuracies, num_classes, class_names


def mode_degradation(cfg, device, baseline_accuracies=None,
                     num_classes=None, class_names=None):
    """
    Step 3: Degradation benchmark.
      Phase A – inference only on degraded test images (no weight updates).
      Phase B – re-train each model on degraded train/val/test data
                (cfg.degradation_epochs), then test on degraded test images.

    cfg.epochs            → used by Step 1 (clean training)   — NOT touched here
    cfg.degradation_epochs→ used only inside Phase B            — 5 by default
    """
    print("\n" + "╔" + "═"*58 + "╗")
    print("║  STEP 3: DEGRADATION BENCHMARK" + " "*27 + "║")
    print(f"║  Phase A: inference only" + " "*33 + "║")
    print(f"║  Phase B: re-train epochs = {cfg.degradation_epochs:<30}║")
    print("╚" + "═"*58 + "╝")

    # If called standalone, rebuild data info
    if num_classes is None or class_names is None:
        _, _, _, num_classes, class_names = build_dataloaders(cfg)

    model_names = get_active_models(cfg)
    models_dict = {}
    for name in model_names:
        models_dict[name] = load_trained_model(name, num_classes, cfg, device)

    if baseline_accuracies is None:
        results_path = os.path.join(cfg.results_dir, "all_models_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                saved = json.load(f)
            baseline_accuracies = {
                m: saved[m]["accuracy"]
                for m in model_names if m in saved
            }
        else:
            baseline_accuracies = {m: 1.0 for m in model_names}
            print("[WARN] No baseline accuracies found; using 1.0 as base.")

    run_degradation_benchmark(
        models_dict,
        baseline_accuracies,
        class_names,
        cfg,
        device,
        num_classes=num_classes,
    )


def mode_interpret(cfg, device, model_name_filter=None,
                   num_classes=None, class_names=None):
    """Step 4: Generate interpretability visualisations."""
    print("\n" + "╔" + "═"*58 + "╗")
    print("║  STEP 4: INTERPRETABILITY MAPS" + " "*27 + "║")
    print("╚" + "═"*58 + "╝")

    _, _, test_loader, num_classes_loaded, class_names_loaded = \
        build_dataloaders(cfg)

    if num_classes is None:
        num_classes = num_classes_loaded
    if class_names is None:
        class_names = class_names_loaded

    model_names = get_active_models(cfg)
    if model_name_filter:
        model_names = [m for m in model_names if m == model_name_filter]

    models_dict = {
        name: load_trained_model(name, num_classes, cfg, device)
        for name in model_names
    }

    run_interpretability(models_dict, test_loader, class_names, cfg, device)


def mode_ablation(cfg, device):
    """Step 5: Data scarcity ablation experiment (cfg.ablation_epochs)."""
    print("\n" + "╔" + "═"*58 + "╗")
    print("║  STEP 5: DATA SCARCITY ABLATION" + " "*26 + "║")
    print(f"║  Model: {cfg.ablation_model:<50}║")
    print(f"║  Epochs per fraction: {cfg.ablation_epochs:<36}║")
    print("╚" + "═"*58 + "╝")

    ablation_results, threshold_results = run_ablation(cfg, device)
    return ablation_results, threshold_results


def mode_single(cfg, device, model_name: str):
    """Train and evaluate a single model (clean data, cfg.epochs)."""
    train_loader, val_loader, test_loader, num_classes, class_names = \
        build_dataloaders(cfg)

    model    = get_model(model_name, num_classes, cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[{model_name}] Parameters: {n_params:,}")

    history, best_val_acc, ckpt_path = train_model(
        model, train_loader, val_loader, cfg, model_name, device
    )
    model   = load_trained_model(model_name, num_classes, cfg, device)
    metrics = evaluate_model(
        model, test_loader, class_names, model_name, cfg, device
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_results_table(all_metrics: dict):
    families = {
        "CNN": ["glyphnet", "resnet50", "efficientnet_b3"],
        "RNN": ["cnn_lstm"],
        "ViT": ["vit_b16", "deit_small"],
    }
    header = (f"{'Model':22s}  {'Family':6s}  {'Acc':7s}  "
              f"{'Prec':7s}  {'Rec':7s}  {'F1':7s}")
    print(header)
    print("-"*70)
    for family, names in families.items():
        for name in names:
            if name not in all_metrics:
                continue
            m = all_metrics[name]
            print(f"  {name:20s}  {family:6s}  "
                  f"{m['accuracy']:.4f}  "
                  f"{m['macro_precision']:.4f}  "
                  f"{m['macro_recall']:.4f}  "
                  f"{m['macro_f1']:.4f}")
        print()


def _print_family_summary(all_metrics: dict):
    families = {
        "CNN": ["glyphnet", "resnet50", "efficientnet_b3"],
        "RNN": ["cnn_lstm"],
        "ViT": ["vit_b16", "deit_small"],
    }
    print("\nFAMILY-LEVEL SUMMARY (Average)")
    print("-"*50)
    for family, names in families.items():
        present = [all_metrics[n] for n in names if n in all_metrics]
        if not present:
            continue
        avg_acc = sum(m["accuracy"] for m in present) / len(present)
        avg_f1  = sum(m["macro_f1"]  for m in present) / len(present)
        best_model = max(
            ((n, all_metrics[n]["accuracy"]) for n in names if n in all_metrics),
            key=lambda x: x[1]
        )
        print(f"  {family}: Avg Acc={avg_acc:.4f}  Avg F1={avg_f1:.4f}  "
              f"Best={best_model[0]} ({best_model[1]:.4f})")


def _get_family(model_name: str) -> str:
    for fam, names in MODEL_FAMILIES.items():
        if model_name in names:
            return fam
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hieroglyph Benchmark: CNN vs RNN vs ViT"
    )
    parser.add_argument(
        "--mode", type=str, default="train_all",
        choices=["train_all", "degradation", "interpret",
                 "ablation", "all", "single"],
        help="Which experiment to run"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (for --mode single or --mode interpret)"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Override data directory from config"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of clean training epochs"
    )
    parser.add_argument(
        "--degradation_epochs", type=int, default=None,
        help="Override number of degradation re-training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size"
    )
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    cfg = get_config()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.epochs:
        cfg.epochs = args.epochs
    if args.degradation_epochs:
        cfg.degradation_epochs = args.degradation_epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size

    device = get_device(cfg.device)
    make_dirs(cfg)

    print("\n" + "━"*60)
    print("  Hieroglyph Deep Learning Benchmark")
    print("  Samruddhi Adhikary | UCE2024006")
    print(f"  Device              : {device}")
    print(f"  Data                : {cfg.data_dir}")
    print(f"  Mode                : {args.mode}")
    print(f"  Clean train epochs  : {cfg.epochs}")
    print(f"  Degradation epochs  : {cfg.degradation_epochs}")
    print(f"  Ablation epochs     : {cfg.ablation_epochs}")
    print("━"*60)

    t_start = time.time()

    # ── Mode dispatch ─────────────────────────────────────────────────────────
    if args.mode == "train_all":
        # Step 1+2 only: clean train → clean test
        mode_train_all(cfg, device)

    elif args.mode == "degradation":
        # Step 3 only: Phase A (inference) + Phase B (re-train + re-test)
        mode_degradation(cfg, device)

    elif args.mode == "interpret":
        mode_interpret(cfg, device, model_name_filter=args.model)

    elif args.mode == "ablation":
        mode_ablation(cfg, device)

    elif args.mode == "single":
        if not args.model:
            parser.error("--mode single requires --model <name>")
        mode_single(cfg, device, args.model)

    elif args.mode == "all":
        # ── Full pipeline in correct order ────────────────────────────────────
        print("\n[Pipeline] Running full benchmark pipeline...")
        print(f"[Pipeline] Clean training epochs    : {cfg.epochs}")
        print(f"[Pipeline] Degradation re-train epochs: {cfg.degradation_epochs}")
        print(f"[Pipeline] Ablation epochs          : {cfg.ablation_epochs}")

        # ── Step 1+2: Train on clean data → test on clean data ───────────────
        print("\n[Pipeline] ── Step 1+2: Clean train + test ──")
        all_metrics, baselines, num_classes, class_names = \
            mode_train_all(cfg, device)

        # ── Step 3: Degradation (Phase A inference + Phase B re-train) ───────
        print("\n[Pipeline] ── Step 3: Degradation benchmark ──")
        print(f"            Phase A: inference only on degraded test images")
        print(f"            Phase B: re-train {cfg.degradation_epochs} epochs "
              f"on degraded data, then test")
        mode_degradation(
            cfg, device,
            baseline_accuracies=baselines,
            num_classes=num_classes,
            class_names=class_names,
        )

        # ── Step 4: Interpretability ──────────────────────────────────────────
        print("\n[Pipeline] ── Step 4: Interpretability maps ──")
        mode_interpret(
            cfg, device,
            num_classes=num_classes,
            class_names=class_names,
        )

        # ── Step 5: Ablation ──────────────────────────────────────────────────
        print("\n[Pipeline] ── Step 5: Data scarcity ablation ──")
        mode_ablation(cfg, device)

        print("\n[Pipeline] ✓ All experiments complete.")

    # ── Timing ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    h, m    = divmod(int(elapsed), 3600)
    m, s    = divmod(m, 60)
    print(f"\n  Total time : {h}h {m}m {s}s")
    print(f"  Results    : {cfg.results_dir}")
    print(f"  Plots      : {cfg.plots_dir}\n")


if __name__ == "__main__":
    main()