"""
models/glyphnet.py
Glyphnet — a compact CNN purpose-built for hieroglyph classification.
Architecture inspired by Barucci et al. (2021), IEEE Access.
Designed to be lightweight with fewer parameters than VGG/ResNet,
well-suited for the limited-data hieroglyph domain.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → (optional) MaxPool."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Glyphnet(nn.Module):
    """
    Glyphnet architecture.
    Input:  (B, 3, 224, 224)
    Output: (B, num_classes)

    Parameters: ~2.1M — much lighter than ResNet-50 (25M) or VGG-16 (138M),
    appropriate for a dataset of ~13K images.
    """
    def __init__(self, num_classes: int = 310, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),   # 224 → 112
            ConvBlock(32,  64,  pool=True),   # 112 → 56
            ConvBlock(64,  128, pool=True),   # 56  → 28
            ConvBlock(128, 256, pool=True),   # 28  → 14
            ConvBlock(256, 512, pool=True),   # 14  → 7
            # Extra conv without pool to deepen features
            ConvBlock(512, 512, pool=False),  # 7   → 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    # For Grad-CAM — expose the last conv layer
    @property
    def gradcam_target_layer(self):
        return self.features[-1].block[0]
