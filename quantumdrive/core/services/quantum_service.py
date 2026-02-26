"""
quantum_service.py — Simulated Variational Quantum Classifier (VQC)
====================================================================
Emulates a parameterised quantum circuit that processes the 7-d
feature vector through a series of *rotation gates* and *entangling
layers*, then performs measurement to yield class probabilities.

This is a **software simulation** — no quantum hardware or Qiskit is
required.  Noise injection mimics real-device decoherence.

Architecture (simulated):
    ┌───────────┐   ┌──────────────┐   ┌────────────┐
    │ Encoding  │──▶│ Variational  │──▶│ Measurement │
    │  layer    │   │  layers (×3) │   │  & softmax  │
    └───────────┘   └──────────────┘   └────────────┘

In production this would call a Pennylane / Qiskit Runtime backend.
"""

import math
import random
from typing import List, Tuple

DECISIONS = ["Straight", "Left", "Right", "Stop"]

# Number of qubits (matches feature dim, padded to even)
N_QUBITS = 8
N_LAYERS = 3


def _ry(theta: float) -> float:
    """Simulated RY gate expectation value ⟨Z⟩ = cos(θ)."""
    return math.cos(theta)


def _rz(theta: float) -> float:
    """Simulated RZ gate phase contribution."""
    return math.sin(theta)


def _cnot_entangle(amplitudes: List[float]) -> List[float]:
    """
    Simulated entangling layer: neighbouring-qubit interaction.
    Mixes adjacent amplitudes (toy model of CNOT cascade).
    """
    n = len(amplitudes)
    out = amplitudes[:]
    for i in range(0, n - 1, 2):
        avg = (out[i] + out[i + 1]) / 2.0
        diff = (out[i] - out[i + 1]) / 2.0
        out[i] = avg + 0.1 * diff
        out[i + 1] = avg - 0.1 * diff
    return out


class QuantumService:
    """
    Simulated Variational Quantum Classifier.

    Parameters
    ----------
    noise : bool
        When True, depolarising noise is injected after each layer.
    noise_strength : float
        Standard deviation of Gaussian noise added to amplitudes.
    """

    def __init__(self, noise: bool = True, noise_strength: float = 0.04):
        self.noise = noise
        self.noise_strength = noise_strength
        self._enabled = True

        # Pre-initialised variational parameters (would be trained)
        random.seed(42)
        self._params: List[List[float]] = [
            [random.uniform(-math.pi, math.pi) for _ in range(N_QUBITS)]
            for _ in range(N_LAYERS)
        ]
        random.seed()  # re-randomise global state

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self, on: bool) -> None:
        self._enabled = on

    def decide(self, features: List[float]) -> Tuple[str, float]:
        """
        Run simulated quantum circuit and return (decision, confidence).

        Steps:
            1. Amplitude encoding of features onto N_QUBITS qubits.
            2. N_LAYERS variational rotation + entanglement layers.
            3. Optional noise injection.
            4. Softmax readout over 4 measurement outcomes.
        """
        # ── Step 1: Encoding layer ───────────────────────────────────
        # Pad features to N_QUBITS
        padded = (features + [0.0] * N_QUBITS)[:N_QUBITS]
        amplitudes = [_ry(f * math.pi) for f in padded]

        # ── Step 2 & 3: Variational + entangling layers ─────────────
        for layer_params in self._params:
            # Rotation
            amplitudes = [
                _ry(a * p) + _rz(a + p)
                for a, p in zip(amplitudes, layer_params)
            ]
            # Entanglement
            amplitudes = _cnot_entangle(amplitudes)
            # Noise
            if self.noise:
                amplitudes = [
                    a + random.gauss(0, self.noise_strength)
                    for a in amplitudes
                ]

        # ── Step 4: Measurement → 4 class logits ────────────────────
        # Map 8 qubit outputs to 4 decision classes (pair-sum)
        logits = [
            amplitudes[i] + amplitudes[i + 1]
            for i in range(0, min(8, len(amplitudes)), 2)
        ]
        # Ensure exactly 4
        logits = (logits + [0.0] * 4)[:4]

        # Softmax
        mx = max(logits)
        exps = [math.exp(v - mx) for v in logits]
        total = sum(exps)
        probs = [e / total for e in exps]

        best_idx = probs.index(max(probs))
        confidence = round(probs[best_idx], 4)

        return DECISIONS[best_idx], confidence


# ─── Module-level singleton ─────────────────────────────────────────────────
quantum_service = QuantumService()
