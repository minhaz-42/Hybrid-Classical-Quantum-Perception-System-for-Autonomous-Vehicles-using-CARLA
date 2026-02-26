"""
QuantumDrive — REST Serializers
================================
DRF serializers for API payloads.
"""

from rest_framework import serializers
from .models import VehicleState, DecisionLog, SimulationSession, WeatherCondition


class VehicleStateSerializer(serializers.ModelSerializer):
    """Serialises a single telemetry frame."""

    class Meta:
        model = VehicleState
        fields = [
            "id", "steering_angle", "throttle", "brake",
            "lane_offset", "curvature", "detected_sign",
            "decision", "confidence_score", "speed_kmh",
            "weather", "model_type", "timestamp",
        ]


class DecisionLogSerializer(serializers.ModelSerializer):
    """Serialises classical/quantum decision audit entries."""

    class Meta:
        model = DecisionLog
        fields = [
            "id", "model_type", "decision", "confidence",
            "steering_error", "lane_deviation", "collision_flag",
            "weather", "timestamp",
        ]


class SimulationSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = SimulationSession
        fields = ["id", "status", "quantum_enabled", "started_at", "stopped_at"]


class WeatherChangeSerializer(serializers.Serializer):
    """Validates incoming weather change requests."""
    weather = serializers.ChoiceField(
        choices=[
            "Clear", "Cloudy", "Rain", "Heavy Rain", "Fog", "Night", "Sunset",
        ]
    )
