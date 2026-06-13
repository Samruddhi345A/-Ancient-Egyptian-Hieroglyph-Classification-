"""
dataset.py  –  Data loading, augmentation, splitting, and degradation transforms.
"""
import os
import random
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image, ImageFilter
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transform(image_size: int = 224,
                        hflip: bool = True,
                        rotation: int = 30,
                        brightness: float = 0.3,
                        contrast: float = 0.3) -> transforms.Compose:
    """Augmentation pipeline used during training."""
    aug_list = [
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5 if hflip else 0.0),
        transforms.RandomRotation(degrees=rotation),
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(aug_list)


def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic transform used for validation, testing, interpretability."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Degradation Transforms
# ─────────────────────────────────────────────────────────────────────────────

class GaussianBlurDeg:
    """Simulate stone erosion — increasingly blurry."""
    radii = {1: 1.5, 2: 3.0, 3: 5.0}

    def __init__(self, severity: int):
        self.radius = self.radii[severity]

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))


class OcclusionDeg:
    """Random rectangular patches (simulates physical damage)."""
    sizes = {1: 0.10, 2: 0.20, 3: 0.35}

    def __init__(self, severity: int):
        self.frac = self.sizes[severity]

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).copy()
        h, w = arr.shape[:2]
        n_patches = severity_to_patches(self.frac)
        ph = int(h * self.frac / n_patches)
        pw = int(w * self.frac / n_patches)
        for _ in range(n_patches):
            y = random.randint(0, h - ph)
            x = random.randint(0, w - pw)
            arr[y:y+ph, x:x+pw] = 0
        return Image.fromarray(arr)


def severity_to_patches(frac: float) -> int:
    if frac <= 0.12:
        return 1
    elif frac <= 0.22:
        return 2
    return 3


class ContrastReductionDeg:
    """Reduce contrast (simulate faded inscriptions)."""
    factors = {1: 0.7, 2: 0.5, 3: 0.3}

    def __init__(self, severity: int):
        self.factor = self.factors[severity]

    def __call__(self, img: Image.Image) -> Image.Image:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(self.factor)


class GaussianNoiseDeg:
    """Additive Gaussian noise."""
    stds = {1: 0.05, 2: 0.12, 3: 0.25}

    def __init__(self, severity: int):
        self.std = self.stds[severity]

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, self.std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


DEGRADATION_MAP = {
    "gaussian_blur": GaussianBlurDeg,
    "occlusion": OcclusionDeg,
    "contrast_reduction": ContrastReductionDeg,
    "gaussian_noise": GaussianNoiseDeg,
}


def get_degradation_transform(deg_type: str, severity: int,
                               image_size: int = 224) -> transforms.Compose:
    """Returns the eval transform with a degradation applied before normalisation."""
    deg_cls = DEGRADATION_MAP[deg_type]
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        deg_cls(severity),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_degradation_train_transform(deg_type: str, severity: int,
                                    image_size: int = 224,
                                    hflip: bool = True,
                                    rotation: int = 30,
                                    brightness: float = 0.3,
                                    contrast: float = 0.3) -> transforms.Compose:
    """
    Augmented training transform with degradation applied BEFORE normalisation.
    Used when re-training models on degraded data (Phase B of degradation benchmark).
    Order:
        Resize → RandomCrop → Flip → Rotate → ColorJitter → Degrade → ToTensor → Normalize
    """
    deg_cls = DEGRADATION_MAP[deg_type]
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5 if hflip else 0.0),
        transforms.RandomRotation(degrees=rotation),
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
        deg_cls(severity),                          # ← degradation injected here
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def filter_small_classes(dataset: datasets.ImageFolder,
                          min_samples: int) -> datasets.ImageFolder:
    """Remove classes that have fewer than min_samples images."""
    from collections import Counter
    counts = Counter(dataset.targets)
    valid_classes = {cls for cls, cnt in counts.items() if cnt >= min_samples}
    mask = [i for i, t in enumerate(dataset.targets) if t in valid_classes]

    # Remap class indices to be contiguous
    old_to_new = {old: new for new, old in enumerate(sorted(valid_classes))}
    dataset.samples = [(path, old_to_new[t]) for path, t in dataset.samples
                       if t in valid_classes]
    dataset.targets = [old_to_new[t] for t in dataset.targets if t in valid_classes]
    old_classes = dataset.classes
    dataset.classes = [old_classes[i] for i in sorted(valid_classes)]
    dataset.class_to_idx = {c: i for i, c in enumerate(dataset.classes)}
    return dataset


def stratified_split(dataset: datasets.ImageFolder,
                      train_r: float, val_r: float,
                      seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified 80/10/10 split that respects class balance.
    Returns (train_indices, val_indices, test_indices).
    """
    from collections import defaultdict
    rng = random.Random(seed)

    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in class_indices.items():
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_r))
        n_val = max(1, int(n * val_r))
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    return train_idx, val_idx, test_idx


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader, int, List[str]]:
    """
    Build train / val / test DataLoaders from cfg.
    Returns: (train_loader, val_loader, test_loader, num_classes, class_names)
    """
    # Load full dataset with eval transform first (for splitting)
    base_dataset = datasets.ImageFolder(cfg.data_dir,
                                        transform=get_eval_transform(cfg.image_size))
    base_dataset = filter_small_classes(base_dataset, cfg.min_samples_per_class)
    num_classes = len(base_dataset.classes)
    class_names = base_dataset.classes

    train_idx, val_idx, test_idx = stratified_split(
        base_dataset, cfg.train_ratio, cfg.val_ratio
    )

    # Train subset gets augmented transform
    train_dataset = datasets.ImageFolder(
        cfg.data_dir,
        transform=get_train_transform(cfg.image_size, cfg.aug_hflip,
                                      cfg.aug_rotation, cfg.aug_brightness_jitter,
                                      cfg.aug_contrast_jitter)
    )
    # Apply same class filtering
    train_dataset = filter_small_classes(train_dataset, cfg.min_samples_per_class)

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(base_dataset, val_idx),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(base_dataset, test_idx),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    print(f"[Dataset] Classes: {num_classes} | "
          f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    return train_loader, val_loader, test_loader, num_classes, class_names


def build_degraded_loader(cfg, deg_type: str, severity: int) -> DataLoader:
    """
    Returns a test-only loader with degradation applied (Phase A).
    Same indices as the clean test split — only the transform differs.
    """
    base_dataset = datasets.ImageFolder(
        cfg.data_dir,
        transform=get_degradation_transform(deg_type, severity, cfg.image_size)
    )
    base_dataset = filter_small_classes(base_dataset, cfg.min_samples_per_class)
    _, _, test_idx = stratified_split(base_dataset, cfg.train_ratio, cfg.val_ratio)
    return DataLoader(
        Subset(base_dataset, test_idx),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )


def build_degraded_retrain_loaders(
    cfg, deg_type: str, severity: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader) where ALL three splits
    have the specified degradation applied. Used for Phase B of the degradation
    benchmark — re-training models on corrupted data then testing on corrupted data.

    - train_loader : degraded augmentation transform  (shuffle=True)
    - val_loader   : degraded eval transform          (shuffle=False)
    - test_loader  : degraded eval transform          (shuffle=False)

    The train/val/test index split is identical to the clean split so that
    results are directly comparable.
    """
    # ── Degraded eval transform (val + test) ──────────────────────────────────
    deg_eval_transform = get_degradation_transform(deg_type, severity, cfg.image_size)

    base_eval = datasets.ImageFolder(cfg.data_dir, transform=deg_eval_transform)
    base_eval = filter_small_classes(base_eval, cfg.min_samples_per_class)

    train_idx, val_idx, test_idx = stratified_split(
        base_eval, cfg.train_ratio, cfg.val_ratio
    )

    val_loader = DataLoader(
        Subset(base_eval, val_idx),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(base_eval, test_idx),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    # ── Degraded augmentation transform (train) ───────────────────────────────
    deg_train_transform = get_degradation_train_transform(
        deg_type, severity, cfg.image_size,
        cfg.aug_hflip, cfg.aug_rotation,
        cfg.aug_brightness_jitter, cfg.aug_contrast_jitter,
    )
    base_train = datasets.ImageFolder(cfg.data_dir, transform=deg_train_transform)
    base_train = filter_small_classes(base_train, cfg.min_samples_per_class)

    train_loader = DataLoader(
        Subset(base_train, train_idx),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )

    print(f"[DegDataset] {deg_type} S{severity} | "
          f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    return train_loader, val_loader, test_loader


def build_ablation_loader(cfg, fraction: float) -> Tuple[DataLoader, DataLoader]:
    """Returns train/val loaders using only `fraction` of the training data."""
    base_eval = datasets.ImageFolder(cfg.data_dir,
                                     transform=get_eval_transform(cfg.image_size))
    base_eval = filter_small_classes(base_eval, cfg.min_samples_per_class)
    train_idx, val_idx, _ = stratified_split(
        base_eval, cfg.train_ratio, cfg.val_ratio
    )

    # Sub-sample training indices
    rng = random.Random(99)
    n_keep = max(1, int(len(train_idx) * fraction))
    train_idx_sub = rng.sample(train_idx, n_keep)

    base_aug = datasets.ImageFolder(
        cfg.data_dir,
        transform=get_train_transform(cfg.image_size, cfg.aug_hflip,
                                      cfg.aug_rotation, cfg.aug_brightness_jitter,
                                      cfg.aug_contrast_jitter)
    )
    base_aug = filter_small_classes(base_aug, cfg.min_samples_per_class)

    train_loader = DataLoader(
        Subset(base_aug, train_idx_sub),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(base_eval, val_idx),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, val_loader