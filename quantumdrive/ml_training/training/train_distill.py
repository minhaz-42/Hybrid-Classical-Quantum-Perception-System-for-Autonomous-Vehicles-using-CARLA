"""
QuantumDrive — Knowledge Distillation Training Pipeline
=========================================================
Production-grade training script with:
    - Mixed-precision (AMP) training
    - Gradient accumulation
    - Checkpoint saving / resume
    - TensorBoard logging
    - Learning rate scheduling (cosine annealing + warmup)
    - Early stopping
    - Google Colab GPU compatibility

Usage:
    python -m ml_training.training.train_distill \\
        --teacher resnet50 \\
        --student mobilenet \\
        --dataset_root ./data/driving \\
        --epochs 100 \\
        --batch_size 64 \\
        --temperature 4.0 \\
        --alpha 0.3 \\
        --beta 0.7 \\
        --output_dir ./checkpoints
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, random_split
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise ImportError(
        "PyTorch with TensorBoard required: "
        "pip install torch torchvision tensorboard"
    )

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml_training.models.teacher_model import create_teacher
from ml_training.models.student_model import create_student
from ml_training.distillation.distillation_loss import DistillationLoss
from ml_training.dataset.driving_dataset import (
    DrivingPerceptionDataset,
    SyntheticDrivingDataset,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Training Engine
# ═══════════════════════════════════════════════════════════════════════════════

class DistillationTrainer:
    """
    Knowledge Distillation training engine.

    Supports:
        - Mixed precision training (FP16)
        - Gradient accumulation for large effective batch sizes
        - Cosine annealing with linear warmup
        - Checkpoint save/resume
        - TensorBoard metrics logging
        - Early stopping on validation loss
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"[QuantumDrive] Using device: {self.device}")

        # Output directory
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ─── Models ───
        self.teacher = create_teacher(
            architecture=config["teacher"],
            num_classes=config["num_classes"],
            pretrained=config.get("pretrained", True),
        ).to(self.device)
        self.teacher.eval()  # Teacher is frozen

        self.student = create_student(
            architecture=config["student"],
            num_classes=config["num_classes"],
            pretrained=config.get("pretrained", True),
        ).to(self.device)

        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        compression = teacher_params / max(student_params, 1)
        print(f"[QuantumDrive] Teacher: {config['teacher']} ({teacher_params:,} params)")
        print(f"[QuantumDrive] Student: {config['student']} ({student_params:,} params)")
        print(f"[QuantumDrive] Compression ratio: {compression:.1f}x")

        # ─── Loss ───
        self.criterion = DistillationLoss(
            temperature=config["temperature"],
            alpha=config["alpha"],
            beta=config["beta"],
        )

        # ─── Optimizer ───
        self.optimizer = optim.AdamW(
            self.student.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.01),
        )

        # ─── Scheduler: Linear warmup + Cosine annealing ───
        warmup_epochs = config.get("warmup_epochs", 5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"] - warmup_epochs
        )
        self.warmup_epochs = warmup_epochs
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, total_iters=warmup_epochs
        )

        # ─── Mixed Precision ───
        self.use_amp = config.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_accum_steps = config.get("grad_accum_steps", 1)

        # ─── TensorBoard ───
        log_dir = self.output_dir / "tensorboard" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # ─── State ───
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.patience = config.get("patience", 15)
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    def _get_dataloaders(self):
        """Create or load dataloaders."""
        config = self.config
        dataset_root = config.get("dataset_root")

        if dataset_root and Path(dataset_root).exists():
            full_dataset = DrivingPerceptionDataset(
                root=dataset_root, split="train",
                max_samples=config.get("max_samples"),
            )
        else:
            print("[QuantumDrive] No dataset found. Using synthetic data for training.")
            full_dataset = SyntheticDrivingDataset(
                num_samples=config.get("num_synthetic_samples", 10000)
            )

        # Split into train/val
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config.get("num_workers", 2),
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config.get("num_workers", 2),
            pin_memory=True,
        )
        return train_loader, val_loader

    def _train_one_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.student.train()
        self.teacher.eval()

        running_total = 0.0
        running_hard = 0.0
        running_soft = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # Handle both image datasets and feature datasets
            if isinstance(batch[1], dict):
                inputs = batch[0].to(self.device)
                labels = batch[1]["class"].to(self.device)
            else:
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

            with autocast(enabled=self.use_amp):
                # Teacher forward (no gradient)
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)

                # Student forward
                student_logits = self.student(inputs)

                # Distillation loss
                losses = self.criterion(student_logits, teacher_logits, labels)
                loss = losses["total"] / self.grad_accum_steps

            # Backward
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_total += losses["total"].item()
            running_hard += losses["hard"].item()
            running_soft += losses["soft"].item()
            num_batches += 1

        avg_total = running_total / max(num_batches, 1)
        avg_hard = running_hard / max(num_batches, 1)
        avg_soft = running_soft / max(num_batches, 1)

        return {"total": avg_total, "hard": avg_hard, "soft": avg_soft}

    @torch.no_grad()
    def _validate(self, val_loader) -> Dict[str, float]:
        """Validate the student model."""
        self.student.eval()
        self.teacher.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch in val_loader:
            if isinstance(batch[1], dict):
                inputs = batch[0].to(self.device)
                labels = batch[1]["class"].to(self.device)
            else:
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

            student_logits = self.student(inputs)
            teacher_logits = self.teacher(inputs)

            losses = self.criterion(student_logits, teacher_logits, labels)
            running_loss += losses["total"].item()

            preds = student_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / max(len(val_loader), 1)
        accuracy = correct / max(total, 1)

        return {"loss": avg_loss, "accuracy": accuracy}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "history": self.history,
        }
        path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  ★ New best model saved (val_loss={self.best_val_loss:.4f})")

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            epoch_path = self.output_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, path: str):
        """Resume training from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint.get("history", self.history)
        print(f"[QuantumDrive] Resumed from epoch {self.start_epoch}")

    def train(self):
        """Full training loop."""
        config = self.config
        train_loader, val_loader = self._get_dataloaders()
        print(f"[QuantumDrive] Training: {len(train_loader)} batches/epoch")
        print(f"[QuantumDrive] Validation: {len(val_loader)} batches")
        print(f"[QuantumDrive] Mixed precision: {self.use_amp}")
        print("=" * 70)

        for epoch in range(self.start_epoch, config["epochs"]):
            start = time.time()

            # ─── Train ───
            train_metrics = self._train_one_epoch(train_loader, epoch)

            # ─── Validate ───
            val_metrics = self._validate(val_loader)

            # ─── Schedule LR ───
            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start

            # ─── Log metrics ───
            self.history["train_loss"].append(train_metrics["total"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])

            self.writer.add_scalar("Loss/train_total", train_metrics["total"], epoch)
            self.writer.add_scalar("Loss/train_hard", train_metrics["hard"], epoch)
            self.writer.add_scalar("Loss/train_soft", train_metrics["soft"], epoch)
            self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            self.writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            # ─── Print progress ───
            print(
                f"Epoch [{epoch+1}/{config['epochs']}] "
                f"train_loss={train_metrics['total']:.4f} "
                f"(hard={train_metrics['hard']:.4f} soft={train_metrics['soft']:.4f}) "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"lr={current_lr:.6f} "
                f"[{elapsed:.1f}s]"
            )

            # ─── Checkpoint ───
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best=is_best)

            # ─── Early stopping ───
            if self.patience_counter >= self.patience:
                print(f"[QuantumDrive] Early stopping at epoch {epoch+1}")
                break

        # ─── Save final artifacts ───
        self.writer.close()
        self._save_training_report()
        print("=" * 70)
        print(f"[QuantumDrive] Training complete. Best val_loss: {self.best_val_loss:.4f}")
        print(f"[QuantumDrive] Checkpoints saved to: {self.output_dir}")

    def _save_training_report(self):
        """Save training report as JSON."""
        report = {
            "config": self.config,
            "best_val_loss": self.best_val_loss,
            "final_epoch": len(self.history["train_loss"]),
            "history": self.history,
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
        }
        report_path = self.output_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="QuantumDrive Knowledge Distillation Training"
    )
    parser.add_argument("--teacher", type=str, default="resnet50",
                        choices=["resnet50", "convnext", "vit", "hybrid_cnn_vit"])
    parser.add_argument("--student", type=str, default="mobilenet",
                        choices=["mobilenet", "efficientnet", "tinyvit", "micronet"])
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_synthetic_samples", type=int, default=10000)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/distillation")
    return parser.parse_args()


def main():
    args = parse_args()
    config = vars(args)
    resume_path = config.pop("resume")

    trainer = DistillationTrainer(config)
    if resume_path:
        trainer.load_checkpoint(resume_path)

    trainer.train()


if __name__ == "__main__":
    main()
