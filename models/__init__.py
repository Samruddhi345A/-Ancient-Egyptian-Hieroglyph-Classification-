"""
models/__init__.py
Central model registry — call get_model() to get any model by name.
"""
from .glyphnet import Glyphnet
from .cnn_models import ResNet50Hieroglyph, EfficientNetB3Hieroglyph
from .rnn_model import CnnLstmHieroglyph
from .vit_models import ViTB16Hieroglyph, DeiTSmallHieroglyph

MODEL_REGISTRY = {
    "glyphnet":         Glyphnet,
    "resnet50":         ResNet50Hieroglyph,
    "efficientnet_b3":  EfficientNetB3Hieroglyph,
    "cnn_lstm":         CnnLstmHieroglyph,
    "vit_b16":          ViTB16Hieroglyph,
    "deit_small":       DeiTSmallHieroglyph,
}

MODEL_FAMILIES = {
    "CNN":  ["glyphnet", "resnet50", "efficientnet_b3"],
    "RNN":  ["cnn_lstm"],
    "ViT":  ["vit_b16", "deit_small"],
}


def get_model(name: str, num_classes: int, cfg=None):
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    if name == "cnn_lstm" and cfg is not None:
        return cls(num_classes=num_classes,
                   lstm_hidden=cfg.lstm_hidden_size,
                   lstm_layers=cfg.lstm_num_layers,
                   lstm_dropout=cfg.lstm_dropout)
    return cls(num_classes=num_classes)
