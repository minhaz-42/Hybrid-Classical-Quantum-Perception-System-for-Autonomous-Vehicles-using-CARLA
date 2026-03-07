"""
QuantumDrive — Model Evaluation Pipeline
==========================================
Comprehensive evaluation with:
    - Per-class accuracy, precision, recall, F1
    - Confusion matrix
    - ROC / AUC curves
    - Inference latency benchmarking
    - Model size comparison (teacher vs student)
    - Export results to JSON
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError:
    raise ImportError("PyTorch required: pip install torch")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml_training.models.teacher_model import create_teacher
from ml_training.models.student_model import create_student
from ml_training.dataset.driving_dataset import (
    DrivingPerceptionDataset,
    SyntheticDrivingDataset,
)


class ModelEvaluator:
    """
    Evaluates perception models and compares teacher vs student performance.
    """

    CLASS_NAMES = [
        "speed_limit", "stop", "yield", "traffic_light",
        "no_entry", "pedestrian_crossing", "turn_left",
        "turn_right", "straight", "roundabout",
    ]

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_classes: int = 10,
    ) -> Dict[str, Any]:
        """
        Run full evaluation: accuracy, per-class metrics, confusion matrix.
        """
        model.eval()
        model.to(self.device)

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        ce_loss = nn.CrossEntropyLoss()

        for batch in dataloader:
            if isinstance(batch[1], dict):
                inputs = batch[0].to(self.device)
                labels = batch[1]["class"].to(self.device)
            else:
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

            logits = model(inputs)
            loss = ce_loss(logits, labels)
            total_loss += loss.item()

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        # Overall accuracy
        accuracy = (all_preds == all_labels).mean()
        avg_loss = total_loss / max(len(dataloader), 1)

        # Per-class metrics
        per_class = {}
        for c in range(num_classes):
            mask_true = all_labels == c
            mask_pred = all_preds == c
            tp = (mask_true & mask_pred).sum()
            fp = (~mask_true & mask_pred).sum()
            fn = (mask_true & ~mask_pred).sum()

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            class_name = self.CLASS_NAMES[c] if c < len(self.CLASS_NAMES) else f"class_{c}"
            per_class[class_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(mask_true.sum()),
            }

        # Confusion matrix
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(all_labels, all_preds):
            confusion[t][p] += 1

        # ROC data (per class, one-vs-rest)
        roc_data = {}
        for c in range(num_classes):
            class_probs = all_probs[:, c]
            class_labels = (all_labels == c).astype(int)
            # Sort by descending probability
            order = np.argsort(-class_probs)
            sorted_labels = class_labels[order]
            # Compute TPR / FPR at various thresholds
            tpr_list, fpr_list = [0.0], [0.0]
            tp, fp = 0, 0
            total_pos = sorted_labels.sum()
            total_neg = len(sorted_labels) - total_pos
            for label in sorted_labels:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
                tpr_list.append(tp / max(total_pos, 1))
                fpr_list.append(fp / max(total_neg, 1))

            # AUC via trapezoidal rule
            auc = np.trapz(tpr_list, fpr_list)
            class_name = self.CLASS_NAMES[c] if c < len(self.CLASS_NAMES) else f"class_{c}"
            roc_data[class_name] = {
                "auc": float(auc),
                "fpr": fpr_list[::max(len(fpr_list) // 50, 1)],
                "tpr": tpr_list[::max(len(tpr_list) // 50, 1)],
            }

        return {
            "accuracy": float(accuracy),
            "loss": float(avg_loss),
            "per_class": per_class,
            "confusion_matrix": confusion.tolist(),
            "roc": roc_data,
            "num_samples": len(all_labels),
        }

    def benchmark_latency(
        self,
        model: nn.Module,
        input_shape: tuple = (1, 3, 224, 224),
        num_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """Benchmark model inference latency."""
        model.eval()
        model.to(self.device)
        dummy = torch.randn(*input_shape).to(self.device)

        # Warmup
        for _ in range(warmup):
            model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "fps": float(1000 / np.mean(latencies)),
        }

    def model_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Get model statistics: params, FLOPs estimate, size."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimate model size in MB
        param_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

        buffer_size_mb = sum(
            b.numel() * b.element_size() for b in model.buffers()
        ) / (1024 * 1024)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": float(param_size_mb + buffer_size_mb),
        }

    def compare_teacher_student(
        self,
        teacher: nn.Module,
        student: nn.Module,
        dataloader: DataLoader,
        num_classes: int = 10,
    ) -> Dict[str, Any]:
        """Compare teacher and student model performance."""
        print("[QuantumDrive] Evaluating teacher model...")
        teacher_metrics = self.evaluate(teacher, dataloader, num_classes)
        teacher_latency = self.benchmark_latency(teacher)
        teacher_stats = self.model_stats(teacher)

        print("[QuantumDrive] Evaluating student model...")
        student_metrics = self.evaluate(student, dataloader, num_classes)
        student_latency = self.benchmark_latency(student)
        student_stats = self.model_stats(student)

        return {
            "teacher": {
                **teacher_metrics,
                "latency": teacher_latency,
                "model_stats": teacher_stats,
            },
            "student": {
                **student_metrics,
                "latency": student_latency,
                "model_stats": student_stats,
            },
            "comparison": {
                "accuracy_gap": teacher_metrics["accuracy"] - student_metrics["accuracy"],
                "speedup": teacher_latency["mean_ms"] / max(student_latency["mean_ms"], 0.01),
                "compression_ratio": teacher_stats["total_params"] / max(student_stats["total_params"], 1),
                "size_reduction": teacher_stats["model_size_mb"] / max(student_stats["model_size_mb"], 0.01),
            },
        }


def main():
    """CLI evaluation entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="QuantumDrive Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="student",
                        choices=["teacher", "student"])
    parser.add_argument("--architecture", type=str, default="mobilenet")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    # Create model
    if args.model_type == "teacher":
        model = create_teacher(args.architecture, args.num_classes, pretrained=False)
    else:
        model = create_student(args.architecture, args.num_classes, pretrained=False)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_key = f"{args.model_type}_state_dict"
    if state_key in ckpt:
        model.load_state_dict(ckpt[state_key])
    else:
        model.load_state_dict(ckpt)

    # Create dataloader
    if args.dataset_root:
        dataset = DrivingPerceptionDataset(args.dataset_root, split="val")
    else:
        dataset = SyntheticDrivingDataset(num_samples=2000)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(model, dataloader, args.num_classes)
    latency = evaluator.benchmark_latency(model)
    stats = evaluator.model_stats(model)

    results["latency"] = latency
    results["model_stats"] = stats

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[QuantumDrive] Results saved to {args.output}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Latency: {latency['mean_ms']:.2f}ms ({latency['fps']:.0f} FPS)")
    print(f"  Model size: {stats['model_size_mb']:.1f} MB")


if __name__ == "__main__":
    main()
