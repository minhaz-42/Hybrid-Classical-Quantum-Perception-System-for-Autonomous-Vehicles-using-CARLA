"""
QuantumDrive — Dataset Writer
===============================
Exports generated data to standard formats (COCO, YOLO).
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class COCOWriter:
    """Export dataset in COCO format."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir) / "coco_format"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, metadata_path: str):
        """Convert QuantumDrive metadata to COCO format."""
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        categories = [
            {"id": i, "name": name, "supercategory": "traffic_sign"}
            for i, name in enumerate(metadata.get("classes", []))
        ]

        images = []
        annotations = []
        ann_id = 0

        for frame in metadata.get("frames", []):
            img_id = frame["frame_id"]
            images.append({
                "id": img_id,
                "file_name": frame["filename"],
                "width": metadata["dataset_info"]["image_size"][0],
                "height": metadata["dataset_info"]["image_size"][1],
            })

            if frame.get("traffic_sign") and frame["sign_class_id"] >= 0:
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": frame["sign_class_id"],
                    "bbox": [100, 80, 60, 60],  # placeholder
                    "area": 3600,
                    "iscrowd": 0,
                    "attributes": {
                        "weather": frame.get("weather", "clear"),
                        "lane_offset": frame.get("lane_offset", 0),
                    },
                })
                ann_id += 1

        coco = {
            "info": {
                "description": "QuantumDrive Synthetic Driving Dataset",
                "version": "1.0",
                "year": 2026,
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [{"id": 1, "name": "Research Use", "url": ""}],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }

        output_path = self.output_dir / "annotations.json"
        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"[QuantumDrive] COCO annotations exported to {output_path}")
        return str(output_path)


class YOLOWriter:
    """Export dataset in YOLO format."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir) / "yolo_format"
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)

    def export(self, metadata_path: str):
        """Convert QuantumDrive metadata to YOLO format."""
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        classes = metadata.get("classes", [])
        img_w, img_h = metadata["dataset_info"]["image_size"]

        for frame in metadata.get("frames", []):
            frame_id = frame["frame_id"]
            label_file = self.output_dir / "labels" / f"{frame_id:06d}.txt"

            lines = []
            if frame.get("traffic_sign") and frame["sign_class_id"] >= 0:
                # YOLO format: class x_center y_center width height (normalized)
                cx, cy = 0.5, 0.35  # placeholder center
                w, h = 0.1, 0.12   # placeholder size
                lines.append(f"{frame['sign_class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            with open(label_file, "w") as f:
                f.write("\n".join(lines))

        # Write classes file
        with open(self.output_dir / "classes.txt", "w") as f:
            for cls in classes:
                f.write(f"{cls}\n")

        # Write data.yaml for YOLO training
        data_yaml = {
            "train": str(self.output_dir / "images"),
            "val": str(self.output_dir / "images"),
            "nc": len(classes),
            "names": classes,
        }
        with open(self.output_dir / "data.yaml", "w") as f:
            for key, value in data_yaml.items():
                f.write(f"{key}: {value}\n")

        print(f"[QuantumDrive] YOLO labels exported to {self.output_dir / 'labels'}")
        return str(self.output_dir)


class DatasetWriter:
    """Unified dataset writer supporting multiple formats."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.coco = COCOWriter(output_dir)
        self.yolo = YOLOWriter(output_dir)

    def export_all(self, metadata_path: str):
        """Export dataset in all supported formats."""
        self.coco.export(metadata_path)
        self.yolo.export(metadata_path)
        print(f"[QuantumDrive] All exports complete.")
