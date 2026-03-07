"""
QuantumDrive — REST Serializers
================================
DRF serializers for API payloads.
"""

from rest_framework import serializers
from .models import (
    VehicleState, DecisionLog, SimulationSession, WeatherCondition,
    Experiment, ExperimentResult, TrainedModel, ResearchMetric,
)


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


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW SERIALIZERS — Experiments & Research
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentSerializer(serializers.ModelSerializer):
    results_count = serializers.SerializerMethodField()

    class Meta:
        model = Experiment
        fields = [
            "id", "name", "experiment_type", "model_name", "dataset_name",
            "status", "config", "tags", "hardware_info",
            "created_at", "updated_at", "started_at", "finished_at",
            "runtime_seconds", "results_count",
        ]

    def get_results_count(self, obj):
        return obj.results.count()


class ExperimentResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExperimentResult
        fields = ["id", "experiment", "epoch", "metric_name", "metric_value", "timestamp"]


class TrainedModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainedModel
        fields = [
            "id", "name", "architecture", "model_type", "experiment",
            "checkpoint_path", "num_params", "model_size_mb",
            "accuracy", "inference_ms", "metadata", "created_at",
        ]


class ResearchMetricSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResearchMetric
        fields = ["id", "experiment", "metric_type", "data", "description", "created_at"]


class PredictRequestSerializer(serializers.Serializer):
    """Validates /api/predict requests."""
    features = serializers.ListField(
        child=serializers.FloatField(), required=False,
        help_text="7-dimensional feature vector"
    )
    model_type = serializers.ChoiceField(
        choices=["classical", "quantum"], default="quantum", required=False
    )
