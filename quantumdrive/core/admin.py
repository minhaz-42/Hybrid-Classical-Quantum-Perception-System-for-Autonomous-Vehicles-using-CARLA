"""
QuantumDrive — Admin Configuration
====================================
Register all core models with the Django admin for easy inspection.
"""

from django.contrib import admin
from .models import SimulationSession, WeatherCondition, VehicleState, DecisionLog


@admin.register(SimulationSession)
class SimulationSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "status", "quantum_enabled", "started_at", "stopped_at")
    list_filter = ("status",)


@admin.register(WeatherCondition)
class WeatherConditionAdmin(admin.ModelAdmin):
    list_display = ("name", "updated_at")


@admin.register(VehicleState)
class VehicleStateAdmin(admin.ModelAdmin):
    list_display = (
        "id", "steering_angle", "throttle", "brake",
        "lane_offset", "detected_sign", "decision",
        "confidence_score", "timestamp",
    )
    list_filter = ("decision", "detected_sign")
    date_hierarchy = "timestamp"


@admin.register(DecisionLog)
class DecisionLogAdmin(admin.ModelAdmin):
    list_display = (
        "id", "model_type", "decision", "confidence",
        "steering_error", "collision_flag", "weather", "timestamp",
    )
    list_filter = ("model_type", "collision_flag", "weather")
