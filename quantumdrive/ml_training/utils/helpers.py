"""
QuantumDrive — ML Training Utilities
======================================
Shared utilities for training and evaluation pipelines.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

try:
    import torch
except ImportError:
    raise ImportError("PyTorch required: pip install torch")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def model_size_mb(model: torch.nn.Module) -> float:
    """Compute model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class MetricsLogger:
    """Simple metrics logger that saves to JSON."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, list] = {}

    def log(self, step: int, **kwargs):
        """Log metrics for a given step."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append({"step": step, "value": float(value)})

    def save(self, filename: str = "metrics.json"):
        """Save all logged metrics to JSON."""
        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def get_best(self, metric: str, mode: str = "min") -> Optional[Dict]:
        """Get the step with the best value for a given metric."""
        if metric not in self.metrics:
            return None
        entries = self.metrics[metric]
        if mode == "min":
            return min(entries, key=lambda x: x["value"])
        return max(entries, key=lambda x: x["value"])
