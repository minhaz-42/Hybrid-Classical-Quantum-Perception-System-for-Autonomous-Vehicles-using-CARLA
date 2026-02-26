"""
classical_service.py — Classical MLP Decision Engine
=====================================================
A lightweight multi-layer perceptron (MLP) that maps the 7-d perception
feature vector to one of four driving decisions:

    0 = Straight   1 = Left   2 = Right   3 = Stop

The weights are hand-crafted to produce plausible behaviour without
requiring a training dataset.  In production this module would load a
serialised PyTorch / ONNX model.
"""

import math
import random
from typing import List, Tuple

# ─── Decision labels ────────────────────────────────────────────────────────
DECISIONS = ["Straight", "Left", "Right", "Stop"]

# ─── Hardcoded weight matrices (small MLP: 7 → 8 → 4) ──────────────────────
# These were designed to give sensible reactions to the feature layout
# described in perception_service.py.

W1: List[List[float]] = [
    [ 0.3, -0.5,  0.1,  0.8, -0.2,  0.4,  0.1],
    [-0.4,  0.6, -0.3, -0.7,  0.5, -0.1,  0.2],
    [ 0.2,  0.1,  0.7, -0.3,  0.6,  0.8, -0.4],
    [-0.1, -0.2,  0.5,  0.4, -0.3,  0.9,  0.3],
    [ 0.5,  0.3, -0.6,  0.2,  0.1, -0.5,  0.6],
    [-0.3,  0.4,  0.2,  0.6, -0.7,  0.3, -0.2],
    [ 0.6, -0.1, -0.4, -0.5,  0.8,  0.2,  0.5],
    [ 0.1,  0.7,  0.3,  0.1, -0.4, -0.6, -0.3],
]

B1: List[float] = [0.1, -0.1, 0.2, 0.05, -0.15, 0.1, -0.05, 0.2]

W2: List[List[float]] = [
    [ 0.5, -0.3,  0.2, -0.6,  0.4,  0.1, -0.2,  0.3],
    [-0.4,  0.6, -0.1,  0.3, -0.5,  0.7,  0.2, -0.3],
    [ 0.3, -0.2,  0.5, -0.1,  0.6, -0.4,  0.8, -0.1],
    [-0.2,  0.4, -0.6,  0.7, -0.3,  0.5, -0.1,  0.6],
]

B2: List[float] = [0.1, -0.05, 0.05, 0.15]


def _relu(x: float) -> float:
    return max(0.0, x)


def _softmax(logits: List[float]) -> List[float]:
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _matmul_bias(W: List[List[float]], b: List[float], x: List[float]) -> List[float]:
    """Dense layer: y = Wx + b."""
    return [
        sum(w * xi for w, xi in zip(row, x)) + bi
        for row, bi in zip(W, b)
    ]


class ClassicalService:
    """
    Two-layer MLP classifier for driving decisions.

    Returns:
        Tuple of (decision_label, confidence_score).
    """

    def decide(self, features: List[float]) -> Tuple[str, float]:
        """Run forward pass and return (decision, confidence)."""
        # Hidden layer with ReLU activation
        hidden = [_relu(v) for v in _matmul_bias(W1, B1, features)]

        # Output logits → softmax probabilities
        logits = _matmul_bias(W2, B2, hidden)
        probs = _softmax(logits)

        # Add slight noise to simulate real-world variance
        noisy_probs = [p + random.gauss(0, 0.02) for p in probs]
        total = sum(max(0, p) for p in noisy_probs)
        noisy_probs = [max(0, p) / total for p in noisy_probs]

        best_idx = noisy_probs.index(max(noisy_probs))
        return DECISIONS[best_idx], round(noisy_probs[best_idx], 4)


# ─── Module-level singleton ─────────────────────────────────────────────────
classical_service = ClassicalService()
