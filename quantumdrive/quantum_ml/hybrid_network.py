"""
QuantumDrive — Hybrid Classical-Quantum Network
==================================================
Combines a CNN feature extractor with a Variational Quantum Circuit
for hybrid classical-quantum autonomous driving decisions.

Architecture:
    Camera Image → CNN Backbone → Feature Vector → Quantum Circuit → Decision

This is the core inference model for the QuantumDrive platform.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


if HAS_TORCH:

    class HybridQuantumNetwork(nn.Module):
        """
        Hybrid Classical-Quantum Network for autonomous driving.

        Pipeline:
            1. CNN Backbone (MobileNetV3 / custom) extracts visual features
            2. Feature compression to quantum-compatible dimensions
            3. Quantum circuit processes features
            4. Classical post-processing produces final decisions

        Outputs:
            - steering: Steering angle correction [-1, 1]
            - lane_correction: Lane deviation correction signal
            - decision: Driving decision logits (straight, left, right, stop)
            - confidence: Decision confidence score
        """

        def __init__(
            self,
            n_qubits: int = 8,
            n_layers: int = 3,
            n_classes: int = 4,
            use_quantum: bool = True,
            backbone: str = "mobilenet",
        ):
            super().__init__()
            self.n_qubits = n_qubits
            self.use_quantum = use_quantum

            # ─── CNN Backbone ───
            if backbone == "mobilenet":
                from torchvision import models
                mobilenet = models.mobilenet_v3_small(
                    weights=models.MobileNet_V3_Small_Weights.DEFAULT
                )
                self.backbone = nn.Sequential(*list(mobilenet.children())[:-1])
                cnn_out_dim = 576
            else:
                # Lightweight custom CNN
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                cnn_out_dim = 128

            # ─── Feature Compressor ───
            self.feature_compressor = nn.Sequential(
                nn.Linear(cnn_out_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, n_qubits),
                nn.Tanh(),
            )

            # ─── Quantum Circuit (if using PennyLane) ───
            if use_quantum and HAS_PENNYLANE:
                import math
                dev = qml.device("default.qubit", wires=n_qubits)

                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def quantum_circuit(inputs, weights):
                    qml.AngleEmbedding(inputs, wires=range(n_qubits))
                    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

                self.quantum_circuit = quantum_circuit
                weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
                self.quantum_weights = nn.Parameter(torch.randn(weight_shape) * 0.1)

                processing_dim = n_qubits
            else:
                # Classical replacement for quantum circuit
                self.classical_processor = nn.Sequential(
                    nn.Linear(n_qubits, 32),
                    nn.ReLU(),
                    nn.Linear(32, n_qubits),
                    nn.Tanh(),
                )
                processing_dim = n_qubits

            # ─── Decision Heads ───
            self.steering_head = nn.Sequential(
                nn.Linear(processing_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Tanh(),  # [-1, 1]
            )

            self.lane_head = nn.Sequential(
                nn.Linear(processing_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

            self.decision_head = nn.Sequential(
                nn.Linear(processing_dim, 32),
                nn.ReLU(),
                nn.Linear(32, n_classes),
            )

            self.confidence_head = nn.Sequential(
                nn.Linear(processing_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> dict:
            """
            Forward pass.

            Args:
                x: Input image tensor (B, 3, H, W) or feature vector (B, F)

            Returns:
                dict with 'steering', 'lane_correction', 'decision', 'confidence'
            """
            # Handle both image and feature inputs
            if x.dim() == 4:
                features = self.backbone(x).flatten(1)
            else:
                features = x

            # Compress to quantum dimensions
            import math
            q_features = self.feature_compressor(features) * math.pi

            # Quantum or classical processing
            if self.use_quantum and HAS_PENNYLANE:
                B = q_features.size(0)
                processed = torch.stack([
                    torch.stack(self.quantum_circuit(q_features[i], self.quantum_weights))
                    for i in range(B)
                ])
            else:
                processed = self.classical_processor(q_features)

            # Decision heads
            return {
                "steering": self.steering_head(processed).squeeze(-1),
                "lane_correction": self.lane_head(processed).squeeze(-1),
                "decision": self.decision_head(processed),
                "confidence": self.confidence_head(processed).squeeze(-1),
            }

        def predict(self, x: torch.Tensor) -> dict:
            """Convenience method for inference."""
            self.eval()
            with torch.no_grad():
                out = self.forward(x)
                decision_probs = torch.softmax(out["decision"], dim=-1)
                decision_idx = decision_probs.argmax(dim=-1)
                decisions = ["straight", "turn_left", "turn_right", "stop"]

                return {
                    "steering": out["steering"].item(),
                    "lane_correction": out["lane_correction"].item(),
                    "decision": decisions[decision_idx.item()],
                    "decision_probs": decision_probs.squeeze().tolist(),
                    "confidence": out["confidence"].item(),
                }


# ═══════════════════════════════════════════════════════════════════════════════
# Numpy-based Hybrid Network (fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class HybridQuantumNetworkNumpy:
    """
    Numpy-based hybrid classical-quantum network.
    For use when PyTorch is not available.
    """

    def __init__(self, n_qubits: int = 8, n_features: int = 7, seed: int = 42):
        from quantum_ml.qnn_model import SimulatedQNN

        self.qnn = SimulatedQNN(
            n_qubits=n_qubits,
            n_features=n_features,
            seed=seed,
        )

    def predict(self, features: np.ndarray) -> dict:
        """Predict from raw feature vector."""
        result = self.qnn.predict(features)

        # Add steering and lane correction from decision
        decision = result["decision"]
        steering_map = {"straight": 0.0, "turn_left": -0.5, "turn_right": 0.5, "stop": 0.0}
        result["steering"] = steering_map.get(decision, 0.0)
        result["lane_correction"] = -features[3] * 0.5 if len(features) > 3 else 0.0

        return result
