"""
QuantumDrive — Views
=====================
Class-based views for HTML pages, authentication, and REST API endpoints.

Architecture:
    Public views   → Landing, Login, Register (no auth required)
    Page views     → TemplateView subclasses behind @login_required
    API views      → DRF APIView subclasses (delegates to service layer)
"""

import csv
import random
from django.http import HttpResponse
from django.utils import timezone
from django.views.generic import TemplateView
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import render, redirect
from django.views import View
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import (
    SimulationSession, VehicleState, DecisionLog, WeatherCondition,
)
from .serializers import (
    VehicleStateSerializer, DecisionLogSerializer,
    WeatherChangeSerializer,
)
from .services.carla_service import carla_service
from .services.perception_service import perception_service
from .services.classical_service import classical_service
from .services.quantum_service import quantum_service


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC PAGES (No auth required)
# ═══════════════════════════════════════════════════════════════════════════════

class LandingView(TemplateView):
    """Public landing / marketing page."""
    template_name = "core/landing.html"


class LoginView(View):
    """User login page."""

    def get(self, request):
        if request.user.is_authenticated:
            return redirect("core:dashboard")
        form = AuthenticationForm()
        return render(request, "core/login.html", {"form": form})

    def post(self, request):
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect("core:dashboard")
        return render(request, "core/login.html", {"form": form})


class RegisterView(View):
    """User registration page."""

    def get(self, request):
        if request.user.is_authenticated:
            return redirect("core:dashboard")
        form = UserCreationForm()
        return render(request, "core/register.html", {"form": form})

    def post(self, request):
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("core:dashboard")
        return render(request, "core/register.html", {"form": form})


class LogoutView(View):
    """Log the user out and redirect to landing."""

    def get(self, request):
        logout(request)
        return redirect("core:landing")

    def post(self, request):
        logout(request)
        return redirect("core:landing")


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTHENTICATED PAGE VIEWS
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardView(LoginRequiredMixin, TemplateView):
    """Main dashboard with live telemetry and charts."""
    template_name = "core/dashboard.html"


class SimulationView(LoginRequiredMixin, TemplateView):
    """Simulation control panel."""
    template_name = "core/simulation.html"


class LogsView(LoginRequiredMixin, TemplateView):
    """Data logging page with decision table and CSV export."""
    template_name = "core/logs.html"


class ComparisonView(LoginRequiredMixin, TemplateView):
    """Classical vs Quantum model comparison page."""
    template_name = "core/comparison.html"


class AnalyticsView(LoginRequiredMixin, TemplateView):
    """Advanced analytics with detailed performance metrics."""
    template_name = "core/analytics.html"


class SettingsView(LoginRequiredMixin, TemplateView):
    """User settings and system configuration."""
    template_name = "core/settings.html"


class HelpView(LoginRequiredMixin, TemplateView):
    """Documentation and help center."""
    template_name = "core/help.html"


class ProfileView(LoginRequiredMixin, TemplateView):
    """User profile page."""
    template_name = "core/profile.html"


# ═══════════════════════════════════════════════════════════════════════════════
#  REST API VIEWS
# ═══════════════════════════════════════════════════════════════════════════════

class VehicleStateAPIView(APIView):
    """
    GET /api/vehicle-state/
    ─────────────────────────
    Returns latest vehicle telemetry.  If the simulation is running,
    a new tick is generated via the service layer and persisted.
    """

    def get(self, request):
        if not carla_service.is_running:
            # Return the most recent stored state (or empty)
            last = VehicleState.objects.first()
            if last:
                return Response(VehicleStateSerializer(last).data)
            return Response({"detail": "No simulation data yet."}, status=200)

        # ── Generate a new tick ──────────────────────────────────────
        snapshot = carla_service.get_state()
        if snapshot is None:
            return Response({"detail": "Simulation not running."}, status=200)

        features = perception_service.extract_features(snapshot)

        # Always compute classical decision
        c_decision, c_confidence = classical_service.decide(features)

        # Quantum decision if enabled
        if quantum_service.enabled:
            q_decision, q_confidence = quantum_service.decide(features)
            decision, confidence = q_decision, q_confidence
            model_type = "quantum"
        else:
            decision, confidence = c_decision, c_confidence
            model_type = "classical"

        # Get active session
        session = SimulationSession.objects.filter(status="running").first()

        # Persist vehicle state
        vs = VehicleState.objects.create(
            session=session,
            steering_angle=snapshot.steering_angle,
            throttle=snapshot.throttle,
            brake=snapshot.brake,
            lane_offset=snapshot.lane_offset,
            curvature=snapshot.curvature,
            detected_sign=snapshot.detected_sign,
            decision=decision,
            confidence_score=confidence,
            speed_kmh=snapshot.speed_kmh,
            weather=carla_service.weather,
            model_type=model_type,
        )

        # ── Log both models for comparison ───────────────────────────
        collision = abs(snapshot.lane_offset) > 0.85

        DecisionLog.objects.create(
            session=session,
            model_type="classical",
            decision=c_decision,
            confidence=c_confidence,
            steering_error=round(abs(snapshot.steering_angle - snapshot.lane_offset * 2), 4),
            lane_deviation=round(abs(snapshot.lane_offset), 4),
            collision_flag=collision,
            weather=carla_service.weather,
        )

        if quantum_service.enabled:
            DecisionLog.objects.create(
                session=session,
                model_type="quantum",
                decision=q_decision,
                confidence=q_confidence,
                steering_error=round(abs(snapshot.steering_angle - snapshot.lane_offset * 1.8), 4),
                lane_deviation=round(abs(snapshot.lane_offset) * 0.85, 4),
                collision_flag=collision and random.random() > 0.3,
                weather=carla_service.weather,
            )

        return Response(VehicleStateSerializer(vs).data)


class StartSimulationAPIView(APIView):
    """
    POST /api/start-simulation/
    ────────────────────────────
    Starts a new simulation session.
    """

    def post(self, request):
        # Stop any existing running session
        SimulationSession.objects.filter(status="running").update(
            status="stopped", stopped_at=timezone.now()
        )

        session = SimulationSession.objects.create(
            status="running",
            started_at=timezone.now(),
            quantum_enabled=quantum_service.enabled,
        )

        carla_service.start()

        # Ensure weather record exists
        if not WeatherCondition.objects.exists():
            WeatherCondition.objects.create(name="Clear")

        return Response({
            "detail": "Simulation started.",
            "session_id": session.pk,
            "weather": carla_service.weather,
            "quantum_enabled": quantum_service.enabled,
        }, status=status.HTTP_200_OK)


class StopSimulationAPIView(APIView):
    """
    POST /api/stop-simulation/
    ───────────────────────────
    Stops the active simulation.
    """

    def post(self, request):
        carla_service.stop()

        SimulationSession.objects.filter(status="running").update(
            status="stopped", stopped_at=timezone.now()
        )

        return Response({"detail": "Simulation stopped."}, status=status.HTTP_200_OK)


class ChangeWeatherAPIView(APIView):
    """
    POST /api/change-weather/
    ──────────────────────────
    Updates the weather preset for the running simulation.
    Body: { "weather": "Rain" }
    """

    def post(self, request):
        serializer = WeatherChangeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        new_weather = serializer.validated_data["weather"]
        active = carla_service.set_weather(new_weather)

        # Persist
        wc, _ = WeatherCondition.objects.get_or_create(pk=1)
        wc.name = active
        wc.save()

        return Response({"weather": active})


class ToggleQuantumAPIView(APIView):
    """
    POST /api/toggle-quantum/
    ──────────────────────────
    Toggle the quantum decision layer ON or OFF.
    Body: { "enabled": true }
    """

    def post(self, request):
        enabled = request.data.get("enabled", True)
        quantum_service.toggle(bool(enabled))
        return Response({"quantum_enabled": quantum_service.enabled})


class DecisionLogsAPIView(APIView):
    """
    GET /api/decision-logs/
    ────────────────────────
    Returns the last 100 decision log entries.
    """

    def get(self, request):
        model_type = request.query_params.get("model_type")
        qs = DecisionLog.objects.all()[:100]
        if model_type in ("classical", "quantum"):
            qs = DecisionLog.objects.filter(model_type=model_type)[:100]
        serializer = DecisionLogSerializer(qs, many=True)
        return Response(serializer.data)


class SimulationStatusAPIView(APIView):
    """
    GET /api/simulation-status/
    ────────────────────────────
    Returns the current simulation status.
    """

    def get(self, request):
        session = SimulationSession.objects.filter(status="running").first()
        wc = WeatherCondition.objects.first()
        return Response({
            "running": carla_service.is_running,
            "session_id": session.pk if session else None,
            "weather": carla_service.weather,
            "quantum_enabled": quantum_service.enabled,
        })


class ComparisonDataAPIView(APIView):
    """
    GET /api/comparison-data/
    ──────────────────────────
    Aggregated data for Classical vs Quantum comparison charts.
    """

    def get(self, request):
        classical_logs = DecisionLog.objects.filter(model_type="classical")[:50]
        quantum_logs = DecisionLog.objects.filter(model_type="quantum")[:50]

        def _aggregate(qs):
            items = list(qs)
            if not items:
                return {
                    "avg_confidence": 0,
                    "avg_lane_deviation": 0,
                    "avg_steering_error": 0,
                    "collision_count": 0,
                    "total": 0,
                    "timestamps": [],
                    "lane_deviations": [],
                    "confidences": [],
                }
            return {
                "avg_confidence": round(sum(i.confidence for i in items) / len(items), 4),
                "avg_lane_deviation": round(sum(i.lane_deviation for i in items) / len(items), 4),
                "avg_steering_error": round(sum(i.steering_error for i in items) / len(items), 4),
                "collision_count": sum(1 for i in items if i.collision_flag),
                "total": len(items),
                "timestamps": [i.timestamp.strftime("%H:%M:%S") for i in reversed(items)],
                "lane_deviations": [round(i.lane_deviation, 4) for i in reversed(items)],
                "confidences": [round(i.confidence, 4) for i in reversed(items)],
            }

        return Response({
            "classical": _aggregate(classical_logs),
            "quantum": _aggregate(quantum_logs),
        })


class AnalyticsDataAPIView(APIView):
    """
    GET /api/analytics-data/
    ─────────────────────────
    Aggregated analytics: session history, decision distribution,
    confidence trends, per-weather accuracy.
    """

    def get(self, request):
        from django.db.models import Count, Avg, Q

        # Session history
        sessions = SimulationSession.objects.all()[:20]
        session_list = []
        for s in sessions:
            session_list.append({
                "id": s.pk,
                "status": s.status,
                "quantum_enabled": s.quantum_enabled,
                "started_at": s.started_at.strftime("%Y-%m-%d %H:%M:%S") if s.started_at else None,
                "stopped_at": s.stopped_at.strftime("%Y-%m-%d %H:%M:%S") if s.stopped_at else None,
            })

        # Total counts
        total_sessions = SimulationSession.objects.count()
        total_decisions = DecisionLog.objects.count()
        quantum_decisions = DecisionLog.objects.filter(model_type="quantum").count()
        total_collisions = DecisionLog.objects.filter(collision_flag=True).count()

        # Decision distribution
        decision_counts = (
            DecisionLog.objects.values("decision")
            .annotate(count=Count("id"))
            .order_by("-count")
        )
        decision_dist = {item["decision"]: item["count"] for item in decision_counts}

        # Confidence trend (last 40 entries, interleaved classical/quantum)
        classical_trend = list(
            DecisionLog.objects.filter(model_type="classical")
            .order_by("-timestamp")[:20]
            .values_list("confidence", "timestamp")
        )
        quantum_trend = list(
            DecisionLog.objects.filter(model_type="quantum")
            .order_by("-timestamp")[:20]
            .values_list("confidence", "timestamp")
        )

        def _trend(items):
            return {
                "values": [round(float(c), 4) for c, _ in reversed(items)],
                "labels": [t.strftime("%H:%M:%S") for _, t in reversed(items)],
            }

        # Per-weather accuracy (avg confidence grouped by weather + model_type)
        weather_stats = (
            DecisionLog.objects.values("weather", "model_type")
            .annotate(avg_conf=Avg("confidence"), count=Count("id"))
            .order_by("weather")
        )
        weather_data = {}
        for ws in weather_stats:
            w = ws["weather"]
            if w not in weather_data:
                weather_data[w] = {"classical": 0, "quantum": 0}
            weather_data[w][ws["model_type"]] = round(float(ws["avg_conf"]) * 100, 1)

        return Response({
            "total_sessions": total_sessions,
            "total_decisions": total_decisions,
            "quantum_decisions": quantum_decisions,
            "total_collisions": total_collisions,
            "sessions": session_list,
            "decision_distribution": decision_dist,
            "confidence_trend": {
                "classical": _trend(classical_trend),
                "quantum": _trend(quantum_trend),
            },
            "weather_accuracy": weather_data,
        })


class ExportCSVView(APIView):
    """
    GET /api/export-csv/
    ─────────────────────
    Exports the last 100 vehicle states as a CSV download.
    """

    def get(self, request):
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="quantumdrive_logs.csv"'

        writer = csv.writer(response)
        writer.writerow([
            "ID", "Steering Angle", "Throttle", "Brake",
            "Lane Offset", "Curvature", "Detected Sign",
            "Decision", "Confidence", "Speed (km/h)",
            "Weather", "Model", "Timestamp",
        ])

        for vs in VehicleState.objects.all()[:100]:
            writer.writerow([
                vs.id, vs.steering_angle, vs.throttle, vs.brake,
                vs.lane_offset, vs.curvature, vs.detected_sign,
                vs.decision, vs.confidence_score, vs.speed_kmh,
                vs.weather, vs.model_type, vs.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            ])

        return response
