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
