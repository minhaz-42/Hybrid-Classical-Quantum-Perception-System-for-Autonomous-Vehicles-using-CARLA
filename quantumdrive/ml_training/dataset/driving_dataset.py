"""
QuantumDrive — Driving Perception Dataset
==========================================
PyTorch Dataset for autonomous driving perception training.
Supports image classification (traffic signs), lane detection, and segmentation.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
except ImportError:
    raise ImportError(
        "PyTorch and torchvision are required. "
        "Install with: pip install torch torchvision Pillow"
    )


# ─── Default Transforms ──────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─── Driving Perception Dataset ──────────────────────────────────────────────

class DrivingPerceptionDataset(Dataset):
    """
    Dataset for autonomous driving perception tasks.

    Directory layout expected:
        root/
            images/
                000000.png
                000001.png
                ...
            labels.json   ← {filename: {"class": int, "lane_offset": float, ...}}
    """

    CLASSES = [
        "speed_limit", "stop", "yield", "traffic_light",
        "no_entry", "pedestrian_crossing", "turn_left",
        "turn_right", "straight", "roundabout",
    ]
    NUM_CLASSES = len(CLASSES)

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform or (TRAIN_TRANSFORM if split == "train" else VAL_TRANSFORM)

        # Load labels
        labels_path = self.root / "labels.json"
        if labels_path.exists():
            with open(labels_path, "r") as f:
                self.labels = json.load(f)
        else:
            self.labels = {}

        # Collect image paths
        images_dir = self.root / "images"
        if images_dir.exists():
            self.image_files = sorted([
                f.name for f in images_dir.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg")
            ])
        else:
            self.image_files = []

        if max_samples:
            self.image_files = self.image_files[:max_samples]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_name = self.image_files[idx]
        img_path = self.root / "images" / img_name

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load label
        label_data = self.labels.get(img_name, {"class": 0, "lane_offset": 0.0})
        target = {
            "class": torch.tensor(label_data.get("class", 0), dtype=torch.long),
            "lane_offset": torch.tensor(label_data.get("lane_offset", 0.0), dtype=torch.float32),
        }

        return image, target

    @classmethod
    def create_dataloaders(
        cls,
        root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        max_samples: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        train_ds = cls(root, split="train", max_samples=max_samples)
        val_ds = cls(root, split="val", max_samples=max_samples)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        return train_loader, val_loader


# ─── Synthetic Feature Dataset (for quick experiments) ────────────────────────

class SyntheticDrivingDataset(Dataset):
    """
    Generates synthetic 7-d feature vectors + labels for rapid prototyping.
    Mimics the perception service feature extraction pipeline.
    """

    DECISIONS = ["straight", "turn_left", "turn_right", "stop"]

    def __init__(self, num_samples: int = 10000, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.features = rng.randn(num_samples, 7).astype(np.float32)

        # Generate labels from a simple rule-based policy
        labels = []
        for feat in self.features:
            steering, throttle, brake, lane_off, curvature, sign_sev, speed = feat
            if sign_sev > 1.0 or brake > 0.5:
                labels.append(3)  # stop
            elif curvature > 0.5:
                labels.append(1 if lane_off > 0 else 2)  # turn
            else:
                labels.append(0)  # straight
        self.labels = np.array(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.features[idx]),
            torch.tensor(self.labels[idx]),
        )
