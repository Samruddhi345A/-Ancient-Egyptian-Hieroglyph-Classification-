"""
interpretability.py
Multi-architecture interpretability for domain expert validation.
Section 4.1 of the study.

  CNN   → Grad-CAM (Gradient-weighted Class Activation Mapping)
  RNN   → Input Gradient Saliency Maps
  ViT   → Attention Rollout across all transformer layers

All visualisations are saved as PNGs with the original image side-by-side.
"""
import os
from typing import Optional, List

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM (for CNN models)
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for any CNN with a identifiable target convolutional layer.
    Selvarajus et al. (2017). No external library required.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def save_activation(module, input, output):
            self.activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_full_backward_hook(save_gradient)

    def generate(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input_tensor.
        Returns heatmap in [0,1] of shape (H, W).
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot)

        # Global average pool of gradients: (C,)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1).squeeze()  # (H, W)
        cam = torch.clamp(cam, min=0)
        cam = cam - cam.min()
        denom = cam.max()
        if denom > 0:
            cam = cam / denom
        return cam.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Input Gradient Saliency (for RNN / CNN-LSTM)
# ─────────────────────────────────────────────────────────────────────────────

def input_gradient_saliency(model: nn.Module,
                              input_tensor: torch.Tensor,
                              target_class: Optional[int] = None) -> np.ndarray:
    """
    Vanilla input gradient saliency map.
    Gradient of the target class score w.r.t. input pixels.
    Returns saliency in [0,1] of shape (H, W).
    """
    model.eval()
    inp = input_tensor.clone().requires_grad_(True)
    logits = model(inp)

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    model.zero_grad()
    one_hot = torch.zeros_like(logits)
    one_hot[0, target_class] = 1.0
    logits.backward(gradient=one_hot)

    saliency = inp.grad.data.abs().squeeze()  # (3, H, W)
    saliency, _ = saliency.max(dim=0)         # (H, W) — take channel max
    saliency = saliency - saliency.min()
    denom = saliency.max()
    if denom > 0:
        saliency = saliency / denom
    return saliency.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# ViT Attention Rollout
# ─────────────────────────────────────────────────────────────────────────────

def vit_attention_rollout(model: nn.Module,
                           input_tensor: torch.Tensor,
                           discard_ratio: float = 0.9) -> np.ndarray:
    """
    Attention rollout across all ViT transformer blocks.
    Abnar & Zuidema (2020). Works with timm ViT models.
    Returns heatmap (14×14 for ViT-B/16) normalised to [0,1].
    """
    model.eval()
    attention_maps = []

    # Hook into each attention block
    hooks = []
    def make_hook(idx):
        def hook(module, input, output):
            # timm attention modules return (x, attn) in some configs
            # We need to capture the attention weight matrix
            if hasattr(module, "attn_drop"):
                # The attention map is inside the forward pass
                pass
        return hook

    # Use a simplified approach: run with hooks on QKV projections
    # For timm ViT, blocks[i].attn is the attention module
    attn_weights_list = []

    def attn_hook(module, input, output):
        # Compute attention weights from the Q,K outputs
        # This requires access to the intermediate computation
        pass

    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        blocks = model.model.blocks
        for block in blocks:
            if hasattr(block, "attn"):
                attn_module = block.attn

                def make_attn_hook(module_ref):
                    def hook(module, input, output):
                        # Access scaled dot-product attention weights
                        # timm stores them in attn_weights if registered
                        B, N, C = input[0].shape
                        qkv = module.qkv(input[0])
                        qkv = qkv.reshape(B, N, 3, module.num_heads,
                                          C // module.num_heads).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv.unbind(0)
                        scale = (C // module.num_heads) ** -0.5
                        attn = (q @ k.transpose(-2, -1)) * scale
                        attn = attn.softmax(dim=-1)
                        attn_weights_list.append(attn.detach().cpu())
                    return hook

                h = attn_module.register_forward_hook(make_attn_hook(attn_module))
                hooks.append(h)

    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:
        h.remove()

    if not attn_weights_list:
        # Fallback: return uniform map
        return np.ones((14, 14)) / (14 * 14)

    # Rollout
    result = torch.eye(attn_weights_list[0].size(-1))
    for attn in attn_weights_list:
        attn_avg = attn.mean(dim=1)  # average over heads: (B, N, N)
        attn_avg = attn_avg[0]       # first sample: (N, N)
        # Add identity (residual connection)
        attn_aug = attn_avg + torch.eye(attn_avg.size(0))
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
        result = attn_aug @ result

    # CLS token attention to all patches: result[0, 1:] is (num_patches,)
    mask = result[0, 1:]
    h = w = int(mask.size(0) ** 0.5)
    mask = mask.reshape(h, w).numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalised tensor back to uint8 numpy image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu().squeeze() * std + mean
    img = img.permute(1, 2, 0).clamp(0, 1).numpy()
    return (img * 255).astype(np.uint8)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray,
                     alpha: float = 0.5) -> np.ndarray:
    """Overlay a heatmap (H, W) on an RGB image (H, W, 3)."""
    from PIL import Image as PILImage
    h, w = image.shape[:2]
    hmap = PILImage.fromarray((heatmap * 255).astype(np.uint8))
    hmap = hmap.resize((w, h), PILImage.BILINEAR)
    hmap_arr = np.array(hmap) / 255.0
    colormap = plt.cm.jet(hmap_arr)[:, :, :3]  # (H, W, 3) RGB
    overlay = (1 - alpha) * (image / 255.0) + alpha * colormap
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def save_interpretability_figure(image_tensor: torch.Tensor,
                                   heatmap: np.ndarray,
                                   model_name: str,
                                   method_name: str,
                                   true_label: str,
                                   pred_label: str,
                                   save_path: str):
    """Save a 3-panel figure: original | heatmap | overlay."""
    img_np = denormalise(image_tensor)
    overlay = overlay_heatmap(img_np, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"{model_name} — {method_name}\n"
                 f"True: {true_label} | Predicted: {pred_label}",
                 fontsize=12)
    axes[0].imshow(img_np);      axes[0].set_title("Original");  axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet");
    axes[1].set_title(method_name); axes[1].axis("off")
    axes[2].imshow(overlay);    axes[2].set_title("Overlay");   axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main interpretability runner
# ─────────────────────────────────────────────────────────────────────────────

def run_interpretability(models_dict: dict,
                          test_loader,
                          class_names: List[str],
                          cfg,
                          device: torch.device,
                          n_samples: int = 10):
    """
    Generate interpretability visualisations for n_samples test images
    across all models.
    """
    print("\n[Interpretability] Generating visualisations...")
    os.makedirs(cfg.interpretability_dir, exist_ok=True)

    # Collect sample images
    samples = []
    for images, labels in test_loader:
        for i in range(images.size(0)):
            samples.append((images[i:i+1], labels[i].item()))
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break

    for model_name, model in models_dict.items():
        model = model.to(device)
        model.eval()
        family = _get_family(model_name)
        print(f"  Processing {model_name} ({family})...")

        for idx, (img_t, true_lbl) in enumerate(samples):
            img_t = img_t.to(device)
            with torch.no_grad():
                pred = model(img_t).argmax(dim=1).item()

            true_name = class_names[true_lbl] if true_lbl < len(class_names) else str(true_lbl)
            pred_name = class_names[pred] if pred < len(class_names) else str(pred)

            save_base = os.path.join(cfg.interpretability_dir,
                                     f"{model_name}_sample{idx:03d}")

            try:
                if family == "CNN":
                    if hasattr(model, "gradcam_target_layer"):
                        gcam = GradCAM(model, model.gradcam_target_layer)
                    elif hasattr(model, "model") and hasattr(model.model, "layer4"):
                        gcam = GradCAM(model, model.model.layer4[-1])
                    else:
                        continue
                    heatmap = gcam.generate(img_t)
                    save_interpretability_figure(
                        img_t, heatmap, model_name, "Grad-CAM",
                        true_name, pred_name, save_base + "_gradcam.png"
                    )

                elif family == "RNN":
                    heatmap = input_gradient_saliency(model, img_t.clone())
                    save_interpretability_figure(
                        img_t, heatmap, model_name, "Saliency Map",
                        true_name, pred_name, save_base + "_saliency.png"
                    )

                elif family == "ViT":
                    heatmap = vit_attention_rollout(model, img_t)
                    save_interpretability_figure(
                        img_t, heatmap, model_name, "Attention Rollout",
                        true_name, pred_name, save_base + "_attn.png"
                    )
            except Exception as e:
                print(f"    [WARN] {model_name} sample {idx}: {e}")
                continue

    print(f"  Saved to: {cfg.interpretability_dir}")


def _get_family(model_name: str) -> str:
    if model_name in ("glyphnet", "resnet50", "efficientnet_b3"):
        return "CNN"
    elif model_name == "cnn_lstm":
        return "RNN"
    else:
        return "ViT"
