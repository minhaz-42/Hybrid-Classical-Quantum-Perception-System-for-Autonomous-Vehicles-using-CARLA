"""
QuantumDrive — Teacher Models
==============================
Large-capacity perception models for knowledge distillation.
These models serve as the "teacher" in the distillation pipeline.

Supported architectures:
    - ResNet50
    - ConvNeXt-Base
    - ViT-Base/16
    - HybridCNN-ViT (custom)
"""

from typing import Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    from torchvision import models
except ImportError:
    raise ImportError("PyTorch and torchvision required: pip install torch torchvision")


# ─── ResNet50 Teacher ─────────────────────────────────────────────────────────

class ResNet50Teacher(nn.Module):
    """ResNet50 teacher model for perception classification."""

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector before classification head."""
        modules = list(self.backbone.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        return feature_extractor(x).squeeze(-1).squeeze(-1)


# ─── ConvNeXt Teacher ─────────────────────────────────────────────────────────

class ConvNeXtTeacher(nn.Module):
    """ConvNeXt-Base teacher model."""

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        self.backbone = models.convnext_base(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ─── Vision Transformer Teacher ───────────────────────────────────────────────

class ViTTeacher(nn.Module):
    """ViT-Base/16 teacher model."""

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ─── Hybrid CNN-ViT Teacher ──────────────────────────────────────────────────

class HybridCNNViTTeacher(nn.Module):
    """
    Custom Hybrid CNN-ViT teacher model.
    Uses a CNN backbone (ResNet34) for feature extraction,
    then a Transformer encoder for global reasoning.
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        # CNN feature extractor (first layers of ResNet34)
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)
        self.cnn_features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3,
        )  # Output: (B, 256, 14, 14) for 224x224 input

        # Transformer encoder on spatial tokens
        self.proj = nn.Conv2d(256, 512, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.pos_embed = nn.Parameter(torch.randn(1, 197, 512))  # 196 patches + 1 CLS

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # CNN feature extraction
        features = self.cnn_features(x)          # (B, 256, H, W)
        features = self.proj(features)            # (B, 512, H, W)
        H, W = features.shape[2], features.shape[3]
        tokens = features.flatten(2).transpose(1, 2)  # (B, H*W, 512)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, H*W+1, 512)

        # Add positional embeddings (truncate/pad if needed)
        seq_len = tokens.size(1)
        if seq_len <= self.pos_embed.size(1):
            tokens = tokens + self.pos_embed[:, :seq_len, :]
        else:
            pos = torch.nn.functional.interpolate(
                self.pos_embed.transpose(1, 2), size=seq_len, mode="linear"
            ).transpose(1, 2)
            tokens = tokens + pos

        # Transformer
        tokens = self.transformer(tokens)

        # Classify from CLS token
        cls_out = tokens[:, 0]
        return self.classifier(cls_out)


# ─── Factory Function ─────────────────────────────────────────────────────────

TEACHER_REGISTRY: Dict[str, type] = {
    "resnet50": ResNet50Teacher,
    "convnext": ConvNeXtTeacher,
    "vit": ViTTeacher,
    "hybrid_cnn_vit": HybridCNNViTTeacher,
}


def create_teacher(
    architecture: str = "resnet50",
    num_classes: int = 10,
    pretrained: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Factory for creating teacher models.

    Args:
        architecture: One of 'resnet50', 'convnext', 'vit', 'hybrid_cnn_vit'
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
    """
    if architecture not in TEACHER_REGISTRY:
        raise ValueError(
            f"Unknown teacher architecture: {architecture}. "
            f"Available: {list(TEACHER_REGISTRY.keys())}"
        )
    return TEACHER_REGISTRY[architecture](
        num_classes=num_classes, pretrained=pretrained, **kwargs
    )
