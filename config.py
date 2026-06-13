"""
config.py  –  All hyperparameters and paths in one place.
Edit this file to change experimental settings.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Dataset ──────────────────────────────────────────────────────────────
    data_dir: str = "./dataset/images"          # ImageFolder-style root
    image_size: int = 224
    num_workers: int = 0
    min_samples_per_class: int = 7              # drop classes with fewer images

    # ── Splits ───────────────────────────────────────────────────────────────
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # ── Training ─────────────────────────────────────────────────────────────
    batch_size: int = 16
    epochs: int = 10                            # normal training epochs
    lr: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    label_smoothing: float = 0.1

    # ── Augmentation ─────────────────────────────────────────────────────────
    aug_hflip: bool = True
    aug_vflip: bool = False
    aug_rotation: int = 30
    aug_brightness_jitter: float = 0.3
    aug_contrast_jitter: float = 0.3

    # ── Models to run ─────────────────────────────────────────────────────────
    run_glyphnet: bool = True
    run_resnet50: bool = True
    run_efficientnet_b3: bool = True
    run_cnn_lstm: bool = True
    run_vit_b16: bool = True
    run_deit_small: bool = True

    # ── CNN-LSTM specific ─────────────────────────────────────────────────────
    lstm_hidden_size: int = 512
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3

    # ── Degradation benchmark ─────────────────────────────────────────────────
    degradation_types: List[str] = field(default_factory=lambda: [
        "gaussian_blur", "occlusion", "contrast_reduction", "gaussian_noise"
    ])
    degradation_severities: List[int] = field(default_factory=lambda: [1, 2, 3])

    # Separate epoch count used ONLY during degradation re-training
    degradation_epochs: int = 5

    # ── Ablation ──────────────────────────────────────────────────────────────
    ablation_fractions: List[float] = field(default_factory=lambda: [
        0.10, 0.25, 0.50, 0.75, 1.00
    ])
    ablation_model: str = "resnet50"
    ablation_epochs: int = 5

    # ── Output paths ──────────────────────────────────────────────────────────
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./outputs/checkpoints"
    results_dir: str = "./outputs/results"
    plots_dir: str = "./outputs/plots"
    interpretability_dir: str = "./outputs/interpretability"

    # ── Device ───────────────────────────────────────────────────────────────
    device: str = "auto"   # "auto" | "cuda" | "mps" | "cpu"


def get_config() -> Config:
    return Config()