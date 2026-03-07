"""
QuantumDrive — Quantum Model Training
========================================
Training pipeline for the hybrid classical-quantum network.
Supports both PennyLane (real quantum gradients) and simulated backends.

Usage:
    python -m quantum_ml.train_quantum \\
        --n_qubits 8 \\
        --n_layers 3 \\
        --epochs 50 \\
        --batch_size 32 \\
        --output_dir ./checkpoints/quantum
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class QuantumTrainer:
    """Training engine for hybrid quantum-classical models."""

    def __init__(self, config: Dict[str, Any]):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for training: pip install torch")

        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"[QuantumDrive] Quantum training on: {self.device}")

        # Create model
        from quantum_ml.hybrid_network import HybridQuantumNetwork
        self.model = HybridQuantumNetwork(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            n_classes=config["n_classes"],
            use_quantum=config.get("use_quantum", True),
            backbone=config.get("backbone", "custom"),
        ).to(self.device)

        params = sum(p.numel() for p in self.model.parameters())
        print(f"[QuantumDrive] Model params: {params:,}")

        # Loss functions
        self.decision_loss = nn.CrossEntropyLoss()
        self.steering_loss = nn.MSELoss()
        self.lane_loss = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 5e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"]
        )

        # Output
        self.output_dir = Path(config.get("output_dir", "./checkpoints/quantum"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    def _get_dataloaders(self):
        """Create synthetic dataloaders for quantum training."""
        from ml_training.dataset.driving_dataset import SyntheticDrivingDataset

        dataset = SyntheticDrivingDataset(
            num_samples=self.config.get("num_samples", 5000),
            seed=self.config.get("seed", 42),
        )

        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=self.config["batch_size"],
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=self.config["batch_size"])

        return train_loader, val_loader

    def train(self):
        """Full training loop."""
        train_loader, val_loader = self._get_dataloaders()
        best_val_loss = float("inf")

        print(f"[QuantumDrive] Starting quantum training for {self.config['epochs']} epochs")
        print(f"[QuantumDrive] Quantum enabled: {self.config.get('use_quantum', True)}")
        print("=" * 60)

        for epoch in range(self.config["epochs"]):
            start = time.time()

            # Train
            self.model.train()
            train_loss = 0.0
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.decision_loss(outputs["decision"], labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)

            # Validate
            self.model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(features)
                    loss = self.decision_loss(outputs["decision"], labels)
                    val_loss += loss.item()
                    preds = outputs["decision"].argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_loss /= max(len(val_loader), 1)
            val_acc = correct / max(total, 1)

            self.scheduler.step()
            elapsed = time.time() - start

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            print(
                f"Epoch [{epoch+1}/{self.config['epochs']}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f} [{elapsed:.1f}s]"
            )

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.config,
                    "history": self.history,
                }, self.output_dir / "quantum_best.pt")
                print(f"  ★ New best model (val_loss={best_val_loss:.4f})")

        # Save report
        report = {
            "config": self.config,
            "best_val_loss": best_val_loss,
            "history": self.history,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.output_dir / "quantum_training_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print("=" * 60)
        print(f"[QuantumDrive] Quantum training complete. Best val_loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="QuantumDrive Quantum Training")
    parser.add_argument("--n_qubits", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--use_quantum", action="store_true", default=False)
    parser.add_argument("--backbone", type=str, default="custom")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/quantum")
    args = parser.parse_args()

    trainer = QuantumTrainer(vars(args))
    trainer.train()


if __name__ == "__main__":
    main()
