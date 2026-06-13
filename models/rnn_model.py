"""
models/rnn_model.py
CNN-LSTM Hybrid for hieroglyph image classification.

Architecture:
  1. Convolutional feature extractor (shared ResNet-50 stem up to layer3)
     → produces feature maps of shape (B, C, H', W')
  2. Feature maps are sliced row-wise into sequences:
     each row of the feature map = one timestep for the LSTM.
     This models left-to-right / top-to-bottom spatial dependencies —
     conceptually analogous to scene text recognition pipelines.
  3. Two stacked bi-directional LSTM layers process the sequence.
  4. Final hidden state → FC → class logits.

Novelty note (as stated in the study):
  No prior work applies RNN-based architectures directly to hieroglyph
  image feature extraction and classification. This is a methodological
  contribution of this study.
"""
import torch
import torch.nn as nn
from torchvision import models


class CnnLstmHieroglyph(nn.Module):
    """
    CNN-LSTM hybrid.
    Input:  (B, 3, 224, 224)
    Output: (B, num_classes)
    """
    def __init__(self,
                 num_classes: int = 310,
                 lstm_hidden: int = 512,
                 lstm_layers: int = 2,
                 lstm_dropout: float = 0.3):
        super().__init__()

        # ── CNN Feature Extractor ─────────────────────────────────────────────
        # Use ResNet-50 up to layer3 (output: B x 1024 x 14 x 14 for 224 input)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.cnn_stem = nn.Sequential(
            resnet.conv1,    # (B, 64, 112, 112)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # (B, 64, 56, 56)
            resnet.layer1,   # (B, 256, 56, 56)
            resnet.layer2,   # (B, 512, 28, 28)
            resnet.layer3,   # (B, 1024, 14, 14)
        )
        cnn_out_channels = 1024
        # After stem: feature map is (B, 1024, 14, 14)
        # We treat H=14 as sequence length, W*C as feature dim at each timestep
        self.seq_len = 14          # H' after stem
        self.feat_dim = cnn_out_channels * 14  # C * W'

        # Project feature dim down before LSTM (otherwise too large)
        self.input_proj = nn.Linear(self.feat_dim, lstm_hidden)

        # ── Bi-directional LSTM ───────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(lstm_dropout),
            nn.Linear(lstm_hidden * 2, 512),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(lstm_dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # CNN feature extraction: (B, 1024, 14, 14)
        feat = self.cnn_stem(x)

        # Reshape to sequence: (B, seq_len, feat_dim)
        # seq_len = H, feat_dim = C * W
        feat = feat.permute(0, 2, 1, 3)          # (B, H, C, W)
        feat = feat.contiguous().view(B, self.seq_len, -1)  # (B, 14, 1024*14)

        # Project to LSTM input size
        feat = self.input_proj(feat)              # (B, 14, lstm_hidden)

        # LSTM: output is (B, seq_len, hidden*2)
        lstm_out, _ = self.lstm(feat)

        # Use the output of the last timestep
        out = lstm_out[:, -1, :]                  # (B, hidden*2)

        return self.classifier(out)

    # Saliency maps will use input gradients — no special layer needed
    @property
    def saliency_target_layer(self):
        return self.cnn_stem[-1]
