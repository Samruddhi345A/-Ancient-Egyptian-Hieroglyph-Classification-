"""
models/cnn_models.py
ResNet-50 and EfficientNet-B3 wrappers with ImageNet pretraining.
Fine-tuned for hieroglyph classification.
"""
import torch
import torch.nn as nn
from torchvision import models


class ResNet50Hieroglyph(nn.Module):
    """
    ResNet-50 with ImageNet pretraining, final FC replaced for hieroglyphs.
    All layers are unfrozen (full fine-tuning) — appropriate given our
    relatively large dataset (13K images).
    """
    def __init__(self, num_classes: int = 310, dropout: float = 0.4):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def gradcam_target_layer(self):
        return self.model.layer4[-1].conv3


class EfficientNetB3Hieroglyph(nn.Module):
    """
    EfficientNet-B3 with ImageNet pretraining.
    Computationally efficient alternative to ResNet-50.
    Also used as the CNN backbone in the CNN-ViT hybrid proposal (Section 4.4).
    """
    def __init__(self, num_classes: int = 310, dropout: float = 0.4):
        super().__init__()
        backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),                    # ✅ no inplace
            nn.Linear(in_features, num_classes)
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def gradcam_target_layer(self):
        return self.model.features[-1][0][0]        # ✅ correct depth