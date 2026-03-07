"""
QuantumDrive — Knowledge Distillation Loss
============================================
Implements the distillation loss function:

    Total Loss = α * CE(student_logits, labels)
               + β * T² * KL(softmax(student_logits/T) || softmax(teacher_logits/T))

Where:
    T = temperature (higher → softer probability distributions)
    α = weight for hard label loss (cross-entropy)
    β = weight for soft label loss (KL divergence from teacher)

Optional extensions:
    - Feature-level distillation (intermediate layer matching)
    - Attention transfer loss
    - Patience-based distillation
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("PyTorch required: pip install torch")


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation loss combining hard and soft targets.

    Args:
        temperature: Softmax temperature T for softening logits (default: 4.0)
        alpha: Weight for hard-label cross-entropy loss (default: 0.3)
        beta: Weight for soft-label KL divergence loss (default: 0.7)
        label_smoothing: Optional label smoothing for CE loss (default: 0.1)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """
        Compute distillation loss.

        Args:
            student_logits: Raw logits from student model (B, C)
            teacher_logits: Raw logits from teacher model (B, C)
            labels: Ground truth class indices (B,)

        Returns:
            dict with 'total', 'hard', 'soft' loss values
        """
        T = self.temperature

        # ─── Hard loss: student vs ground truth
        hard_loss = self.ce_loss(student_logits, labels)

        # ─── Soft loss: student vs teacher (softened by temperature)
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (T * T)

        # ─── Total loss
        total_loss = self.alpha * hard_loss + self.beta * soft_loss

        return {
            "total": total_loss,
            "hard": hard_loss.detach(),
            "soft": soft_loss.detach(),
        }


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level knowledge distillation.
    Matches intermediate feature representations between teacher and student.

    Uses MSE loss on projected feature vectors.
    """

    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        temperature: float = 4.0,
        alpha: float = 0.2,
        beta: float = 0.5,
        gamma: float = 0.3,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.feature_loss = nn.MSELoss()

        # Project student features to teacher feature space
        self.feature_projector = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.ReLU(inplace=True),
            nn.Linear(teacher_dim, teacher_dim),
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> dict:
        """
        Compute feature-level distillation loss.

        Total = α*CE + β*KL + γ*MSE(proj(student_feat), teacher_feat)
        """
        T = self.temperature

        # Hard loss
        hard_loss = self.ce_loss(student_logits, labels)

        # Soft loss
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (T * T)

        # Feature matching loss
        projected = self.feature_projector(student_features)
        feat_loss = self.feature_loss(projected, teacher_features.detach())

        total_loss = (
            self.alpha * hard_loss + self.beta * soft_loss + self.gamma * feat_loss
        )

        return {
            "total": total_loss,
            "hard": hard_loss.detach(),
            "soft": soft_loss.detach(),
            "feature": feat_loss.detach(),
        }


class AttentionTransferLoss(nn.Module):
    """
    Attention Transfer distillation (Zagoruyko & Komodakis, 2017).
    Matches spatial attention maps between teacher and student.
    """

    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def attention_map(self, features: torch.Tensor) -> torch.Tensor:
        """Compute spatial attention map from feature tensor (B, C, H, W)."""
        return F.normalize(features.pow(self.p).mean(1).flatten(1), dim=1)

    def forward(
        self,
        student_features: list,
        teacher_features: list,
    ) -> torch.Tensor:
        """
        Compute attention transfer loss across multiple layers.

        Args:
            student_features: List of feature maps from student layers
            teacher_features: List of feature maps from teacher layers
        """
        loss = 0.0
        for sf, tf in zip(student_features, teacher_features):
            s_att = self.attention_map(sf)
            t_att = self.attention_map(tf)
            loss += (s_att - t_att).pow(2).mean()
        return loss / len(student_features)
