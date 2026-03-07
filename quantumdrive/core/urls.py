"""
QuantumDrive — Core URL Routing
=================================
Maps public pages, authenticated pages, and REST API endpoints.

Public routes:
    /                    → Landing page
    /login/              → Login
    /register/           → Registration
    /logout/             → Logout

Authenticated routes:
    /dashboard/          → Dashboard
    /simulation/         → Simulation Control
    /logs/               → Data Logging
    /comparison/         → Model Comparison
    /analytics/          → Analytics
    /settings/           → Settings
    /help/               → Help Center
    /profile/            → User Profile

API routes:
    /api/vehicle-state/       GET
    /api/start-simulation/    POST
    /api/stop-simulation/     POST
    /api/change-weather/      POST
    /api/toggle-quantum/      POST
    /api/decision-logs/       GET
    /api/simulation-status/   GET
    /api/comparison-data/     GET
    /api/export-csv/          GET
"""

from django.urls import path
from . import views

app_name = "core"

urlpatterns = [
    # ── Public Pages ─────────────────────────────────────────────────────
    path("", views.LandingView.as_view(), name="landing"),
    path("login/", views.LoginView.as_view(), name="login"),
    path("register/", views.RegisterView.as_view(), name="register"),
    path("logout/", views.LogoutView.as_view(), name="logout"),

    # ── Authenticated Pages ──────────────────────────────────────────────
    path("dashboard/", views.DashboardView.as_view(), name="dashboard"),
    path("simulation/", views.SimulationView.as_view(), name="simulation"),
    path("logs/", views.LogsView.as_view(), name="logs"),
    path("comparison/", views.ComparisonView.as_view(), name="comparison"),
    path("analytics/", views.AnalyticsView.as_view(), name="analytics"),
    path("settings/", views.SettingsView.as_view(), name="settings"),
    path("help/", views.HelpView.as_view(), name="help"),
    path("profile/", views.ProfileView.as_view(), name="profile"),
    path("research/", views.ResearchDashboardView.as_view(), name="research_dashboard"),

    # ── REST API ─────────────────────────────────────────────────────────
    path("api/vehicle-state/", views.VehicleStateAPIView.as_view(), name="api-vehicle-state"),
    path("api/start-simulation/", views.StartSimulationAPIView.as_view(), name="api-start-simulation"),
    path("api/stop-simulation/", views.StopSimulationAPIView.as_view(), name="api-stop-simulation"),
    path("api/change-weather/", views.ChangeWeatherAPIView.as_view(), name="api-change-weather"),
    path("api/toggle-quantum/", views.ToggleQuantumAPIView.as_view(), name="api-toggle-quantum"),
    path("api/decision-logs/", views.DecisionLogsAPIView.as_view(), name="api-decision-logs"),
    path("api/simulation-status/", views.SimulationStatusAPIView.as_view(), name="api-simulation-status"),
    path("api/comparison-data/", views.ComparisonDataAPIView.as_view(), name="api-comparison-data"),
    path("api/analytics-data/", views.AnalyticsDataAPIView.as_view(), name="api-analytics-data"),
    path("api/export-csv/", views.ExportCSVView.as_view(), name="api-export-csv"),

    # ── Research & ML API ────────────────────────────────────────────────
    path("api/experiments/", views.ExperimentListAPIView.as_view(), name="api-experiments"),
    path("api/experiments/<int:pk>/", views.ExperimentDetailAPIView.as_view(), name="api-experiment-detail"),
    path("api/experiment-results/", views.ExperimentResultsAPIView.as_view(), name="api-experiment-results"),
    path("api/trained-models/", views.TrainedModelListAPIView.as_view(), name="api-trained-models"),
    path("api/research-metrics/", views.ResearchMetricsAPIView.as_view(), name="api-research-metrics"),
    path("api/research-dashboard-data/", views.ResearchDashboardDataAPIView.as_view(), name="api-research-dashboard-data"),
    path("api/predict/", views.PredictAPIView.as_view(), name="api-predict"),
    path("api/run-model/", views.RunModelAPIView.as_view(), name="api-run-model"),
    path("api/upload-frame/", views.UploadFrameAPIView.as_view(), name="api-upload-frame"),
]
