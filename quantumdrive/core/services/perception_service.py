"""
perception_service.py — Feature Extraction Pipeline
=====================================================
Processes raw simulated CARLA telemetry into a normalised feature
vector suitable for both classical and quantum decision models.

In production, this would wrap a CNN/ViT feature extractor operating
on camera frames and LiDAR point clouds.
"""

import math
from typing import Dict, List

from .carla_service import VehicleSnapshot

# ─── Sign → numeric encoding ────────────────────────────────────────────────
SIGN_ENCODING: Dict[str, float] = {
    "None": 0.0,
    "Stop": 1.0,
    "Yield": 0.8,
    "Speed Limit 30": 0.3,
    "Speed Limit 60": 0.5,
    "Speed Limit 90": 0.7,
    "No Entry": 1.0,
    "Right Turn": 0.4,
    "Left Turn": 0.4,
    "Pedestrian Crossing": 0.9,
    "Traffic Light Red": 1.0,
    "Traffic Light Green": 0.1,
}


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


class PerceptionService:
    """
    Transforms a ``VehicleSnapshot`` into a fixed-length feature vector.

    Features (7-d):
        0 — normalised steering angle  [-1, 1]
        1 — throttle                   [0, 1]
        2 — brake                      [0, 1]
        3 — lane offset (sigmoid)      [0, 1]
        4 — curvature (sigmoid)        [0, 1]
        5 — sign severity encoding     [0, 1]
        6 — speed (normalised to 160 km/h cap)
    """

    FEATURE_DIM = 7

    def extract_features(self, snapshot: VehicleSnapshot) -> List[float]:
        """Return a 7-dimensional feature vector from raw telemetry."""
        return [
            snapshot.steering_angle,
            snapshot.throttle,
            snapshot.brake,
            _sigmoid(snapshot.lane_offset * 5.0),
            _sigmoid(snapshot.curvature * 100.0),
            SIGN_ENCODING.get(snapshot.detected_sign, 0.0),
            min(snapshot.speed_kmh / 160.0, 1.0),
        ]


# ─── Module-level singleton ─────────────────────────────────────────────────
perception_service = PerceptionService()
