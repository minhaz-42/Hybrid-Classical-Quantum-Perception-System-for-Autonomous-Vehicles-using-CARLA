"""
QuantumDrive — Quantum Neural Network Model
==============================================
Variational Quantum Circuit (VQC) for autonomous driving decision-making.

Architecture:
    AngleEmbedding → StronglyEntanglingLayers → Measurement

Uses PennyLane for quantum circuit simulation with a PyTorch interface.
Falls back to a numpy-based simulator if PennyLane is not available.

Predicts:
    - Lane deviation correction
    - Steering adjustment
    - Driving decision (straight, left, right, stop)
"""

import math
from typing import List, Optional, Tuple

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


# ═══════════════════════════════════════════════════════════════════════════════
# PennyLane-based QNN (production)
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_PENNYLANE and HAS_TORCH:

    class PennyLaneQNN(nn.Module):
        """
        Variational Quantum Classifier using PennyLane.

        Architecture:
            1. AngleEmbedding: Encodes classical features into qubit rotations
            2. StronglyEntanglingLayers: Parameterized rotation + CNOT entanglement
            3. Measurement: Pauli-Z expectation values on each qubit

        Args:
            n_qubits: Number of qubits (should >= input feature dimension)
            n_layers: Number of variational layers
            n_features: Input feature dimension
            n_outputs: Number of output classes/values
        """

        def __init__(
            self,
            n_qubits: int = 8,
            n_layers: int = 4,
            n_features: int = 7,
            n_outputs: int = 4,
        ):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.n_features = n_features
            self.n_outputs = n_outputs

            # PennyLane device
            dev = qml.device("default.qubit", wires=n_qubits)

            # Define quantum circuit
            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(inputs, weights):
                # Angle embedding of input features
                qml.AngleEmbedding(inputs[:n_qubits], wires=range(n_qubits))

                # Strongly entangling layers
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

                # Measure all qubits
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            self.circuit = circuit

            # Trainable quantum weights
            weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
            self.weights = nn.Parameter(
                torch.randn(weight_shape) * 0.1
            )

            # Pre-processing: project features to n_qubits dimensions
            self.pre_net = nn.Sequential(
                nn.Linear(n_features, n_qubits),
                nn.Tanh(),  # Bound to [-1, 1] → scale to [0, π]
            )

            # Post-processing: map qubit measurements to output
            self.post_net = nn.Sequential(
                nn.Linear(n_qubits, 32),
                nn.ReLU(),
                nn.Linear(32, n_outputs),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Args:
                x: Input features (B, n_features)

            Returns:
                Output logits (B, n_outputs)
            """
            B = x.size(0)

            # Pre-process to qubit-compatible inputs
            q_in = self.pre_net(x) * math.pi  # Scale to [0, π]

            # Run quantum circuit for each sample
            q_out = torch.stack([
                torch.stack(self.circuit(q_in[i], self.weights))
                for i in range(B)
            ])  # (B, n_qubits)

            # Post-process to final output
            return self.post_net(q_out)


# ═══════════════════════════════════════════════════════════════════════════════
# Numpy-based QNN simulator (fallback when PennyLane unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

class SimulatedQNN:
    """
    Simulated Quantum Neural Network using numpy.
    Provides the same interface as PennyLaneQNN but uses classical simulation.

    This implements a simplified VQC:
        1. Angle embedding via RY gates
        2. Entangling layers via parameterized RZ/RY + CNOT
        3. Measurement via Z-expectation value estimation
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 4,
        n_features: int = 7,
        n_outputs: int = 4,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.n_outputs = n_outputs

        rng = np.random.RandomState(seed)

        # Quantum weights: (n_layers, n_qubits, 3) for RZ, RY, RZ rotations
        self.weights = rng.randn(n_layers, n_qubits, 3) * 0.3

        # Pre-processing weights
        self.pre_w = rng.randn(n_features, n_qubits) * 0.5
        self.pre_b = rng.randn(n_qubits) * 0.1

        # Post-processing weights
        self.post_w1 = rng.randn(n_qubits, 32) * 0.3
        self.post_b1 = rng.randn(32) * 0.1
        self.post_w2 = rng.randn(32, n_outputs) * 0.3
        self.post_b2 = rng.randn(n_outputs) * 0.1

    def _ry_gate(self, theta: float) -> np.ndarray:
        """Single-qubit RY rotation matrix."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def _rz_gate(self, theta: float) -> np.ndarray:
        """Single-qubit RZ rotation matrix."""
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)

    def _apply_single_gate(self, state: np.ndarray, gate: np.ndarray, qubit: int):
        """Apply a single-qubit gate to the statevector."""
        n = self.n_qubits
        dim = 2 ** n
        new_state = np.zeros(dim, dtype=complex)

        for i in range(dim):
            bit = (i >> (n - 1 - qubit)) & 1
            partner = i ^ (1 << (n - 1 - qubit))
            if bit == 0:
                new_state[i] += gate[0, 0] * state[i] + gate[0, 1] * state[partner]
            else:
                new_state[i] += gate[1, 0] * state[partner] + gate[1, 1] * state[i]

        return new_state

    def _cnot(self, state: np.ndarray, control: int, target: int):
        """Apply CNOT gate."""
        n = self.n_qubits
        dim = 2 ** n
        new_state = state.copy()

        for i in range(dim):
            ctrl_bit = (i >> (n - 1 - control)) & 1
            if ctrl_bit == 1:
                flipped = i ^ (1 << (n - 1 - target))
                new_state[i], new_state[flipped] = state[flipped], state[i]

        return new_state

    def _measure_z(self, state: np.ndarray, qubit: int) -> float:
        """Compute <Z> expectation value for a qubit."""
        n = self.n_qubits
        dim = 2 ** n
        expval = 0.0

        for i in range(dim):
            bit = (i >> (n - 1 - qubit)) & 1
            sign = 1 - 2 * bit  # +1 for |0>, -1 for |1>
            expval += sign * (np.abs(state[i]) ** 2)

        return float(expval)

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass through simulated quantum circuit.

        Args:
            features: Input features (n_features,)

        Returns:
            Output logits (n_outputs,)
        """
        # Pre-process
        q_in = np.tanh(features @ self.pre_w + self.pre_b) * np.pi

        # Initialize |0...0> state
        dim = 2 ** self.n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        # Angle embedding
        for i in range(min(self.n_qubits, len(q_in))):
            state = self._apply_single_gate(state, self._ry_gate(q_in[i]), i)

        # Strongly entangling layers
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                rz1, ry, rz2 = self.weights[layer, qubit]
                state = self._apply_single_gate(state, self._rz_gate(rz1), qubit)
                state = self._apply_single_gate(state, self._ry_gate(ry), qubit)
                state = self._apply_single_gate(state, self._rz_gate(rz2), qubit)

            # CNOT entangling (ring topology)
            for qubit in range(self.n_qubits):
                target = (qubit + 1) % self.n_qubits
                state = self._cnot(state, qubit, target)

        # Measure
        measurements = np.array([self._measure_z(state, q) for q in range(self.n_qubits)])

        # Post-process
        hidden = np.maximum(0, measurements @ self.post_w1 + self.post_b1)  # ReLU
        output = hidden @ self.post_w2 + self.post_b2

        return output

    def predict(self, features: np.ndarray) -> dict:
        """Predict driving decision from feature vector."""
        logits = self.forward(features)
        probs = np.exp(logits) / np.exp(logits).sum()  # softmax
        decision_idx = int(np.argmax(probs))
        decisions = ["straight", "turn_left", "turn_right", "stop"]

        return {
            "decision": decisions[decision_idx],
            "confidence": float(probs[decision_idx]),
            "probabilities": probs.tolist(),
            "logits": logits.tolist(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_qnn(backend: str = "auto", **kwargs):
    """
    Create a Quantum Neural Network.

    Args:
        backend: "pennylane", "simulated", or "auto"
    """
    if backend == "auto":
        backend = "pennylane" if (HAS_PENNYLANE and HAS_TORCH) else "simulated"

    if backend == "pennylane":
        if not HAS_PENNYLANE or not HAS_TORCH:
            raise ImportError("PennyLane and PyTorch required for pennylane backend")
        return PennyLaneQNN(**kwargs)
    elif backend == "simulated":
        return SimulatedQNN(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
