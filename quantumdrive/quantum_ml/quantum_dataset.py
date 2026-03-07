"""
QuantumDrive — Quantum Dataset
================================
Dataset utilities specific to quantum model training.
Provides normalized feature vectors compatible with quantum angle embedding.
"""

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class QuantumDrivingDataset:
    """
    Dataset for quantum circuit training.
    Features are normalized to [0, π] for angle embedding.
    """

    DECISIONS = ["straight", "turn_left", "turn_right", "stop"]

    def __init__(self, num_samples: int = 5000, n_features: int = 7, seed: int = 42):
        rng = np.random.RandomState(seed)

        # Generate driving features
        self.features = np.zeros((num_samples, n_features), dtype=np.float32)
        self.features[:, 0] = rng.uniform(-1, 1, num_samples)    # steering
        self.features[:, 1] = rng.uniform(0, 1, num_samples)     # throttle
        self.features[:, 2] = rng.uniform(0, 1, num_samples)     # brake
        self.features[:, 3] = rng.normal(0, 0.5, num_samples)    # lane offset
        self.features[:, 4] = rng.uniform(-0.05, 0.05, num_samples)  # curvature
        self.features[:, 5] = rng.choice([0, 0.3, 0.6, 1.0], num_samples)  # sign severity
        self.features[:, 6] = rng.uniform(0, 1, num_samples)     # speed norm

        # Generate labels based on driving rules
        labels = []
        for feat in self.features:
            steering, throttle, brake, lane_off, curvature, sign_sev, speed = feat
            if sign_sev > 0.7 or brake > 0.6:
                labels.append(3)  # stop
            elif abs(curvature) > 0.02 or abs(lane_off) > 0.8:
                labels.append(1 if lane_off > 0 else 2)  # turn
            else:
                labels.append(0)  # straight
        self.labels = np.array(labels, dtype=np.int64)

        # Normalize features to [0, π] for quantum embedding
        self.features_normalized = np.zeros_like(self.features)
        for i in range(n_features):
            col = self.features[:, i]
            min_val, max_val = col.min(), col.max()
            if max_val - min_val > 1e-8:
                self.features_normalized[:, i] = (col - min_val) / (max_val - min_val) * np.pi
            else:
                self.features_normalized[:, i] = 0.0

    def __len__(self):
        return len(self.features)

    def get_batch(self, start: int, end: int):
        """Get a batch of samples."""
        return self.features_normalized[start:end], self.labels[start:end]

    def get_classical_vs_quantum_split(self):
        """Split data for classical vs quantum comparison experiments."""
        n = len(self.features)
        mid = n // 2
        return {
            "classical_train": (self.features[:mid], self.labels[:mid]),
            "quantum_train": (self.features_normalized[:mid], self.labels[:mid]),
            "classical_test": (self.features[mid:], self.labels[mid:]),
            "quantum_test": (self.features_normalized[mid:], self.labels[mid:]),
        }


if HAS_TORCH:
    class QuantumDrivingTorchDataset(Dataset):
        """PyTorch-compatible quantum driving dataset."""

        def __init__(self, num_samples: int = 5000, seed: int = 42):
            ds = QuantumDrivingDataset(num_samples=num_samples, seed=seed)
            self.features = torch.tensor(ds.features_normalized, dtype=torch.float32)
            self.labels = torch.tensor(ds.labels, dtype=torch.long)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
