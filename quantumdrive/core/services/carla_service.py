"""
carla_service.py — CARLA Data Stream Simulator
================================================
Generates realistic mock telemetry data that would normally come from
a CARLA autonomous driving simulator. Produces lane geometry, vehicle
kinematics, detected traffic signs, and weather conditions.

In production this module would wrap the CARLA Python API client.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

# ─── Constants ───────────────────────────────────────────────────────────────
TRAFFIC_SIGNS = [
    "None", "Stop", "Yield", "Speed Limit 30", "Speed Limit 60",
    "Speed Limit 90", "No Entry", "Right Turn", "Left Turn",
    "Pedestrian Crossing", "Traffic Light Red", "Traffic Light Green",
]

WEATHER_PRESETS = [
    "Clear", "Cloudy", "Rain", "Heavy Rain", "Fog", "Night", "Sunset",
]


@dataclass
class VehicleSnapshot:
    """Single frame of simulated vehicle telemetry."""
    steering_angle: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    lane_offset: float = 0.0
    curvature: float = 0.0
    detected_sign: str = "None"
    speed_kmh: float = 0.0


class CARLAService:
    """
    Simulates a CARLA data stream.

    Maintains internal time-step counter to produce smoothly varying
    telemetry rather than purely random noise.
    """

    def __init__(self):
        self._tick: int = 0
        self._weather: str = "Clear"
        self._running: bool = False
        self._base_offset: float = 0.0

    # ── Public API ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin the simulation loop."""
        self._running = True
        self._tick = 0
        self._base_offset = random.uniform(-0.3, 0.3)

    def stop(self) -> None:
        """Halt the simulation loop."""
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def set_weather(self, weather: str) -> str:
        """Change the weather preset. Returns the active weather."""
        if weather in WEATHER_PRESETS:
            self._weather = weather
        return self._weather

    @property
    def weather(self) -> str:
        return self._weather

    def get_state(self) -> Optional[VehicleSnapshot]:
        """
        Generate one tick of simulated CARLA data.

        Returns ``None`` when the simulation is not running.
        """
        if not self._running:
            return None

        self._tick += 1
        t = self._tick * 0.1  # virtual seconds

        # Smoothly varying lane offset (sinusoidal + drift)
        lane_offset = (
            self._base_offset
            + 0.4 * math.sin(0.3 * t)
            + random.gauss(0, 0.05)
        )

        # Curvature follows a slower wave
        curvature = 0.01 * math.sin(0.1 * t) + random.gauss(0, 0.002)

        # Steering reacts to offset (simple P-controller mock)
        steering_angle = max(-1.0, min(1.0, -2.0 * lane_offset + random.gauss(0, 0.05)))

        # Throttle / brake depend on whether a stop sign is ahead
        detected_sign = random.choices(
            TRAFFIC_SIGNS,
            weights=[50, 5, 3, 4, 4, 4, 2, 5, 5, 5, 6, 7],
        )[0]

        if "Stop" in detected_sign or "Red" in detected_sign:
            throttle = max(0.0, 0.15 + random.gauss(0, 0.05))
            brake = min(1.0, 0.7 + random.gauss(0, 0.1))
        else:
            throttle = min(1.0, 0.6 + 0.2 * math.sin(0.2 * t) + random.gauss(0, 0.05))
            brake = max(0.0, random.gauss(0, 0.03))

        speed_kmh = max(0.0, throttle * 120 - brake * 80 + random.gauss(0, 2))

        # Slowly drift the base offset to simulate lane wander
        self._base_offset += random.gauss(0, 0.01)
        self._base_offset = max(-1.0, min(1.0, self._base_offset))

        return VehicleSnapshot(
            steering_angle=round(steering_angle, 4),
            throttle=round(max(0, throttle), 4),
            brake=round(max(0, brake), 4),
            lane_offset=round(lane_offset, 4),
            curvature=round(curvature, 6),
            detected_sign=detected_sign,
            speed_kmh=round(speed_kmh, 2),
        )


# ─── Module-level singleton ─────────────────────────────────────────────────
carla_service = CARLAService()
