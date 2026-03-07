"""
QuantumDrive — Inference Service
===================================
Connects trained models to the CARLA simulation pipeline.

Flow:
    CARLA Camera → Perception Model → Lane Detection
    → Quantum Decision Model → Vehicle Control

Also provides API-compatible inference for the Django REST endpoints.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class InferenceService:
    """
    Production inference service for QuantumDrive.

    Handles:
        - Model loading and caching
        - Image preprocessing
        - Feature extraction
        - Quantum/classical decision making
        - Results postprocessing
    """

    DECISIONS = ["straight", "turn_left", "turn_right", "stop"]

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "student",  # "teacher", "student", "quantum"
        architecture: str = "mobilenet",
        num_classes: int = 10,
        use_quantum: bool = False,
        device: Optional[str] = None,
    ):
        self.model_type = model_type
        self.architecture = architecture
        self.use_quantum = use_quantum
        self.model = None
        self.quantum_model = None

        if HAS_TORCH:
            self.device = torch.device(
                device or (
                    "cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
            )
        else:
            self.device = None

        # Load model if path provided
        if model_path and HAS_TORCH:
            self.load_model(model_path, model_type, architecture, num_classes)

        # Initialize quantum fallback
        if use_quantum:
            from quantum_ml.qnn_model import SimulatedQNN
            self.quantum_model = SimulatedQNN()

        # Image preprocessing
        if HAS_TORCH:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def load_model(
        self,
        model_path: str,
        model_type: str = "student",
        architecture: str = "mobilenet",
        num_classes: int = 10,
    ):
        """Load a trained model from checkpoint."""
        if not HAS_TORCH:
            print("[InferenceService] PyTorch not available, using fallback")
            return

        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

        if model_type == "teacher":
            from ml_training.models.teacher_model import create_teacher
            self.model = create_teacher(architecture, num_classes, pretrained=False)
        elif model_type == "student":
            from ml_training.models.student_model import create_student
            self.model = create_student(architecture, num_classes, pretrained=False)
        elif model_type == "quantum":
            from quantum_ml.hybrid_network import HybridQuantumNetwork
            self.model = HybridQuantumNetwork(use_quantum=self.use_quantum)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        state_key = f"{model_type}_state_dict"
        if state_key in checkpoint:
            self.model.load_state_dict(checkpoint[state_key])
        elif "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        print(f"[InferenceService] Model loaded: {architecture} ({model_type})")

    def predict_image(self, image: "Image.Image") -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image: PIL Image

        Returns:
            dict with lane_position, steering, confidence, decision
        """
        start = time.perf_counter()

        if self.model is not None and HAS_TORCH:
            # Preprocess
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                if self.model_type == "quantum":
                    output = self.model(tensor)
                    result = {
                        "steering": float(output["steering"].item()),
                        "lane_correction": float(output["lane_correction"].item()),
                        "decision": self.DECISIONS[output["decision"].argmax(-1).item()],
                        "confidence": float(output["confidence"].item()),
                        "decision_probs": F.softmax(output["decision"], dim=-1).squeeze().tolist(),
                    }
                else:
                    logits = self.model(tensor)
                    probs = F.softmax(logits, dim=-1).squeeze()
                    pred_idx = probs.argmax().item()
                    result = {
                        "class_id": pred_idx,
                        "confidence": float(probs[pred_idx].item()),
                        "probabilities": probs.tolist(),
                        "decision": self.DECISIONS[pred_idx % 4],
                        "steering": 0.0,  # Would need lane model
                        "lane_position": 0.0,
                    }
        else:
            # Fallback: simulated inference
            result = self._fallback_inference()

        result["inference_ms"] = (time.perf_counter() - start) * 1000
        return result

    def predict_features(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a pre-extracted feature vector.

        Args:
            features: numpy array of shape (7,) — perception features

        Returns:
            Prediction results
        """
        start = time.perf_counter()

        if self.quantum_model:
            result = self.quantum_model.predict(features)
        elif self.model is not None and HAS_TORCH:
            tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if self.model_type == "quantum":
                    output = self.model(tensor)
                    probs = F.softmax(output["decision"], dim=-1).squeeze()
                    result = {
                        "decision": self.DECISIONS[probs.argmax().item()],
                        "confidence": float(probs.max().item()),
                        "steering": float(output["steering"].item()),
                        "decision_probs": probs.tolist(),
                    }
                else:
                    logits = self.model(tensor)
                    probs = F.softmax(logits, dim=-1).squeeze()
                    result = {
                        "decision": self.DECISIONS[probs.argmax().item() % 4],
                        "confidence": float(probs.max().item()),
                    }
        else:
            result = self._fallback_inference()

        result["inference_ms"] = (time.perf_counter() - start) * 1000
        return result

    def predict_frame_bytes(self, frame_bytes: bytes) -> Dict[str, Any]:
        """
        Run inference on raw image bytes (for API upload endpoint).

        Args:
            frame_bytes: Raw image bytes (PNG/JPEG)

        Returns:
            Prediction results
        """
        if not HAS_PIL:
            return self._fallback_inference()

        import io
        image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        return self.predict_image(image)

    def _fallback_inference(self) -> Dict[str, Any]:
        """Fallback inference when no model is loaded."""
        rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
        probs = rng.dirichlet([2, 1, 1, 0.5])
        idx = int(np.argmax(probs))
        return {
            "decision": self.DECISIONS[idx],
            "confidence": float(probs[idx]),
            "steering": float(rng.uniform(-0.3, 0.3)),
            "lane_position": float(rng.uniform(-0.5, 0.5)),
            "decision_probs": probs.tolist(),
            "fallback": True,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get info about the currently loaded model."""
        info = {
            "model_type": self.model_type,
            "architecture": self.architecture,
            "use_quantum": self.use_quantum,
            "device": str(self.device) if self.device else "cpu",
            "model_loaded": self.model is not None,
        }
        if self.model is not None and HAS_TORCH:
            info["total_params"] = sum(p.numel() for p in self.model.parameters())
        return info


# Module-level singleton
inference_service = InferenceService(use_quantum=False)
