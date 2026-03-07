"""
QuantumDrive — Dataset Loader
===============================
Unified dataset loader with support for COCO and YOLO formats.
Integrates with PyTorch DataLoader.
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
except ImportError:
    raise ImportError("PyTorch and torchvision required: pip install torch torchvision Pillow")


class SyntheticDrivingLoader(Dataset):
    """
    PyTorch Dataset for loading QuantumDrive synthetic data.

    Supports:
        - Direct loading from generator output
        - COCO format annotations
        - YOLO format labels
    """

    def __init__(
        self,
        root: str,
        format: str = "native",  # "native", "coco", "yolo"
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = Path(root)
        self.format = format
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        if format == "native":
            self._load_native()
        elif format == "coco":
            self._load_coco()
        elif format == "yolo":
            self._load_yolo()

    def _load_native(self):
        """Load from QuantumDrive native format."""
        metadata_path = self.root / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.frames = metadata.get("frames", [])
        else:
            # Fallback: list images
            img_dir = self.root / "images"
            self.frames = [
                {"filename": f.name, "class": 0, "lane_offset": 0.0}
                for f in sorted(img_dir.glob("*.png"))
            ] if img_dir.exists() else []

    def _load_coco(self):
        """Load from COCO format annotations."""
        ann_path = self.root / "coco_format" / "annotations.json"
        with open(ann_path, "r") as f:
            coco = json.load(f)

        img_map = {img["id"]: img for img in coco["images"]}
        self.frames = []
        seen_ids = set()

        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in seen_ids:
                seen_ids.add(img_id)
                img = img_map[img_id]
                self.frames.append({
                    "filename": img["file_name"],
                    "class": ann["category_id"],
                    "lane_offset": ann.get("attributes", {}).get("lane_offset", 0),
                })

    def _load_yolo(self):
        """Load from YOLO format."""
        labels_dir = self.root / "yolo_format" / "labels"
        images_dir = self.root / "images"
        self.frames = []

        for label_file in sorted(labels_dir.glob("*.txt")):
            img_name = label_file.stem + ".png"
            if (images_dir / img_name).exists():
                with open(label_file, "r") as f:
                    lines = f.readlines()
                cls = int(lines[0].split()[0]) if lines else 0
                self.frames.append({
                    "filename": img_name,
                    "class": cls,
                    "lane_offset": 0.0,
                })

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        frame = self.frames[idx]
        img_path = self.root / "images" / frame["filename"]

        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            # Placeholder if image missing
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        target = {
            "class": torch.tensor(frame.get("class", 0), dtype=torch.long),
            "lane_offset": torch.tensor(frame.get("lane_offset", 0.0), dtype=torch.float32),
        }

        return image, target

    @classmethod
    def create_loaders(
        cls,
        root: str,
        format: str = "native",
        batch_size: int = 32,
        num_workers: int = 2,
        train_split: float = 0.8,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train/val dataloaders with random split."""
        dataset = cls(root, format=format)

        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        return train_loader, val_loader
