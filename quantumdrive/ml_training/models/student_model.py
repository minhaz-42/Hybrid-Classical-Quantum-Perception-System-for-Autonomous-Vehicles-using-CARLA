"""
QuantumDrive — Student Models
==============================
Lightweight perception models for edge deployment.
These models learn from the teacher via knowledge distillation.

Supported architectures:
    - MobileNetV3-Small
    - EfficientNet-B0 Lite
    - TinyViT (custom lightweight ViT)
    - MicroNet (ultra-lightweight custom CNN)
"""

from typing import Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    from torchvision import models
except ImportError:
    raise ImportError("PyTorch and torchvision required: pip install torch torchvision")


# ─── MobileNetV3 Student ─────────────────────────────────────────────────────

class MobileNetStudent(nn.Module):
    """MobileNetV3-Small student model (~2.5M params)."""

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── EfficientNet-B0 Lite Student ────────────────────────────────────────────

class EfficientNetStudent(nn.Module):
    """EfficientNet-B0 student model (~5.3M params)."""

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── TinyViT Student ──────────────────────────────────────────────────────────

class TinyViTStudent(nn.Module):
    """
    Ultra-lightweight Vision Transformer student.
    ~1.8M params with 6 layers, 4 heads, dim=192.
    """

    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add position embeddings
        x = self.pos_drop(x + self.pos_embed[:, : x.size(1)])

        # Transformer
        x = self.blocks(x)
        x = self.norm(x)

        # Classify from CLS
        return self.head(x[:, 0])

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── MicroNet Student ─────────────────────────────────────────────────────────

class MicroNetStudent(nn.Module):
    """
    Ultra-lightweight CNN student (~0.3M params).
    Designed for real-time edge inference in autonomous vehicles.
    """

    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__()

        def depthwise_sep(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU6(inplace=True),
            )

        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            # Depthwise separable blocks
            depthwise_sep(16, 32, stride=2),
            depthwise_sep(32, 64, stride=2),
            depthwise_sep(64, 64, stride=1),
            depthwise_sep(64, 128, stride=2),
            depthwise_sep(128, 128, stride=1),
            depthwise_sep(128, 256, stride=2),
            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Factory Function ─────────────────────────────────────────────────────────

STUDENT_REGISTRY: Dict[str, type] = {
    "mobilenet": MobileNetStudent,
    "efficientnet": EfficientNetStudent,
    "tinyvit": TinyViTStudent,
    "micronet": MicroNetStudent,
}


def create_student(
    architecture: str = "mobilenet",
    num_classes: int = 10,
    pretrained: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Factory for creating student models.

    Args:
        architecture: One of 'mobilenet', 'efficientnet', 'tinyvit', 'micronet'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for mobilenet/efficientnet)
    """
    if architecture not in STUDENT_REGISTRY:
        raise ValueError(
            f"Unknown student architecture: {architecture}. "
            f"Available: {list(STUDENT_REGISTRY.keys())}"
        )
    return STUDENT_REGISTRY[architecture](
        num_classes=num_classes, pretrained=pretrained, **kwargs
    )
