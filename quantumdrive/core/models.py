"""
QuantumDrive — Core Models
===========================
Database schema for the autonomous driving simulation platform.

Models:
    SimulationSession — tracks start/stop of each run
    WeatherCondition  — active weather preset
    VehicleState      — per-tick telemetry snapshot
    DecisionLog       — classical vs quantum decision audit trail
"""

from django.db import models
from django.utils import timezone


class SimulationSession(models.Model):
    """Represents one continuous simulation run."""

    STATUS_CHOICES = [
        ("running", "Running"),
        ("stopped", "Stopped"),
    ]

    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default="stopped")
    started_at = models.DateTimeField(null=True, blank=True)
    stopped_at = models.DateTimeField(null=True, blank=True)
    quantum_enabled = models.BooleanField(default=True)

    class Meta:
        ordering = ["-started_at"]
        verbose_name = "Simulation Session"

    def __str__(self):
        return f"Session #{self.pk} — {self.status}"


class WeatherCondition(models.Model):
    """Stores the currently active weather preset."""

    WEATHER_CHOICES = [
        ("Clear", "Clear"),
        ("Cloudy", "Cloudy"),
        ("Rain", "Rain"),
        ("Heavy Rain", "Heavy Rain"),
        ("Fog", "Fog"),
        ("Night", "Night"),
        ("Sunset", "Sunset"),
    ]

    name = models.CharField(max_length=20, choices=WEATHER_CHOICES, default="Clear")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Weather Condition"

    def __str__(self):
        return self.name


class VehicleState(models.Model):
    """Single telemetry frame captured during simulation."""

    session = models.ForeignKey(
        SimulationSession,
        on_delete=models.CASCADE,
        related_name="vehicle_states",
        null=True,
        blank=True,
    )
    steering_angle = models.FloatField(default=0.0, help_text="Steering angle [-1, 1]")
    throttle = models.FloatField(default=0.0, help_text="Throttle [0, 1]")
    brake = models.FloatField(default=0.0, help_text="Brake [0, 1]")
    lane_offset = models.FloatField(default=0.0, help_text="Lateral offset from lane centre")
    curvature = models.FloatField(default=0.0, help_text="Road curvature")
    detected_sign = models.CharField(max_length=50, default="None", help_text="Traffic sign label")
    decision = models.CharField(max_length=20, default="Straight", help_text="Driving decision")
    confidence_score = models.FloatField(default=0.0, help_text="Model confidence [0, 1]")
    speed_kmh = models.FloatField(default=0.0, help_text="Vehicle speed in km/h")
    weather = models.CharField(max_length=20, default="Clear")
    model_type = models.CharField(max_length=10, default="quantum", help_text="classical or quantum")
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ["-timestamp"]
        verbose_name = "Vehicle State"

    def __str__(self):
        return f"State @{self.timestamp:%H:%M:%S} — {self.decision} ({self.confidence_score:.2f})"


class DecisionLog(models.Model):
    """
    Audit log comparing classical vs quantum model outputs.
    Used for the Model Comparison page.
    """

    MODEL_CHOICES = [
        ("classical", "Classical"),
        ("quantum", "Quantum"),
    ]

    session = models.ForeignKey(
        SimulationSession,
        on_delete=models.CASCADE,
        related_name="decision_logs",
        null=True,
        blank=True,
    )
    model_type = models.CharField(max_length=10, choices=MODEL_CHOICES)
    decision = models.CharField(max_length=20, default="Straight")
    confidence = models.FloatField(default=0.0)
    steering_error = models.FloatField(default=0.0, help_text="Absolute steering deviation")
    lane_deviation = models.FloatField(default=0.0, help_text="Absolute lane offset")
    collision_flag = models.BooleanField(default=False)
    weather = models.CharField(max_length=20, default="Clear")
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ["-timestamp"]
        verbose_name = "Decision Log"

    def __str__(self):
        return f"{self.model_type} @{self.timestamp:%H:%M:%S} — {self.decision}"


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW MODELS — Experiment Tracking & Research
# ═══════════════════════════════════════════════════════════════════════════════

class Experiment(models.Model):
    """Tracks a single ML experiment run."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    name = models.CharField(max_length=200)
    experiment_type = models.CharField(max_length=50, default="distillation",
        help_text="distillation, quantum, evaluation, comparison")
    model_name = models.CharField(max_length=100, default="",
        help_text="e.g. resnet50→mobilenet")
    dataset_name = models.CharField(max_length=100, default="synthetic")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    config = models.JSONField(default=dict, blank=True,
        help_text="Training hyperparameters and configuration")
    tags = models.JSONField(default=list, blank=True)
    hardware_info = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    runtime_seconds = models.FloatField(default=0)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Experiment"

    def __str__(self):
        return f"{self.name} ({self.status})"


class ExperimentResult(models.Model):
    """Stores metrics and results for an experiment."""

    experiment = models.ForeignKey(
        Experiment, on_delete=models.CASCADE, related_name="results"
    )
    epoch = models.IntegerField(default=0)
    metric_name = models.CharField(max_length=100,
        help_text="e.g. train_loss, val_accuracy, val_loss")
    metric_value = models.FloatField(default=0.0)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["experiment", "epoch", "metric_name"]
        verbose_name = "Experiment Result"

    def __str__(self):
        return f"{self.experiment.name} — {self.metric_name}@{self.epoch}: {self.metric_value:.4f}"


class TrainedModel(models.Model):
    """Registry of trained model artifacts."""

    name = models.CharField(max_length=200)
    architecture = models.CharField(max_length=100,
        help_text="e.g. mobilenet, resnet50, hybrid_qnn")
    model_type = models.CharField(max_length=50, default="student",
        help_text="teacher, student, quantum, hybrid")
    experiment = models.ForeignKey(
        Experiment, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="trained_models")
    checkpoint_path = models.CharField(max_length=500, default="")
    num_params = models.BigIntegerField(default=0)
    model_size_mb = models.FloatField(default=0.0)
    accuracy = models.FloatField(default=0.0)
    inference_ms = models.FloatField(default=0.0, help_text="Mean inference latency in ms")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Trained Model"

    def __str__(self):
        return f"{self.name} ({self.architecture})"


class ResearchMetric(models.Model):
    """Stores research-grade metrics for analytics dashboards."""

    METRIC_TYPES = [
        ("confusion_matrix", "Confusion Matrix"),
        ("roc_curve", "ROC Curve"),
        ("precision_recall", "Precision-Recall"),
        ("distillation_curve", "Distillation Curve"),
        ("attention_map", "Attention Map"),
        ("quantum_comparison", "Quantum Comparison"),
    ]

    experiment = models.ForeignKey(
        Experiment, on_delete=models.CASCADE, related_name="research_metrics",
        null=True, blank=True)
    metric_type = models.CharField(max_length=50, choices=METRIC_TYPES)
    data = models.JSONField(default=dict,
        help_text="Structured metric data (JSON)")
    description = models.TextField(default="", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Research Metric"

    def __str__(self):
        return f"{self.metric_type} — {self.created_at:%Y-%m-%d %H:%M}"
