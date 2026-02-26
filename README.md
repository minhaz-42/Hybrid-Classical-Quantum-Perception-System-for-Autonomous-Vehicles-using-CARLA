# QuantumDrive

Hybrid Classical–Quantum Perception System for Lane, Route, and Traffic-Sign Understanding in Autonomous Vehicles using CARLA.

QuantumDrive is a Django + DRF research platform that combines:
- classical perception/decision signals,
- quantum-enhanced decision logic,
- real-time simulation telemetry,
- interactive analytics and model comparison dashboards.

It provides a full web workflow for autonomous-driving experimentation: start simulation sessions, switch weather, toggle quantum mode, inspect decision logs, and compare classical vs quantum outcomes.

---

## Table of Contents
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Run the App](#run-the-app)
- [API Documentation](#api-documentation)
    - [Authentication](#authentication)
    - [Endpoints](#endpoints)
    - [Example Requests](#example-requests)
- [Data Model Summary](#data-model-summary)
- [Frontend Behavior & Settings](#frontend-behavior--settings)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

- **Live Dashboard:** Real-time steering/throttle/brake/speed/lane metrics with streaming charts.
- **Simulation Control:** Start/stop sessions, weather switching, quantum toggle, and live driving visualization.
- **Decision Logging:** Color-coded decision logs with confidence, lane deviation, steering error, and collision flags.
- **Model Comparison:** Aggregated classical vs quantum performance metrics and chart-based comparison.
- **Analytics:** Session trends, decision distribution, confidence-over-time, and weather-wise model confidence.
- **CSV Export:** Download recent telemetry rows from the API.
- **Authentication:** Login/register/logout flow with protected pages.

---

## System Architecture

```text
Browser UI (Templates + Chart.js + app.js)
                |
                | REST/JSON
                v
Django + DRF (views, serializers, models)
                |
                +--> Classical Service
                +--> Quantum Service
                +--> Perception Service
                +--> CARLA Service (or simulated fallback)
```

---

## Tech Stack

- **Backend:** Python, Django 5.x, Django REST Framework
- **Frontend:** Django Templates, JavaScript (ES6+), Chart.js
- **Database:** SQLite (default development database)
- **Simulation:** CARLA integration through service layer
- **Optional Quantum Extensions:** Qiskit/PennyLane (currently optional/pluggable)

---

## Repository Structure

```text
.
├── manage.py                # Root wrapper (delegates to quantumdrive/manage.py)
├── requirements.txt
├── README.md
└── quantumdrive/
        ├── manage.py            # Django manage entrypoint
        ├── db.sqlite3
        ├── core/
        │   ├── models.py
        │   ├── serializers.py
        │   ├── urls.py
        │   ├── views.py
        │   └── services/
        │       ├── carla_service.py
        │       ├── perception_service.py
        │       ├── classical_service.py
        │       └── quantum_service.py
        ├── quantumdrive/
        │   ├── settings.py
        │   ├── urls.py
        │   ├── asgi.py
        │   └── wsgi.py
        ├── templates/
        └── static/
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- `pip`
- (Optional) CARLA simulator if you want live simulator-backed data

### Installation

```bash
# From project root
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python3 manage.py migrate
python3 manage.py createsuperuser  # optional
```

### Run the App

```bash
# Run from repository root
python3 manage.py runserver 8000
```

Then open:

- `http://127.0.0.1:8000/` (landing page)
- `http://127.0.0.1:8000/dashboard/` (after login)

---

## API Documentation

Base path: `/api/`

### Authentication

- Browser pages are authenticated via Django sessions.
- API calls from the frontend include CSRF token for `POST` endpoints.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/vehicle-state/` | Returns latest vehicle telemetry; when running, generates and stores a fresh tick. |
| POST | `/api/start-simulation/` | Starts a simulation session and returns session status metadata. |
| POST | `/api/stop-simulation/` | Stops active simulation session. |
| POST | `/api/change-weather/` | Changes weather. Body: `{ "weather": "Rain" }` |
| POST | `/api/toggle-quantum/` | Enables/disables quantum mode. Body: `{ "enabled": true }` |
| GET | `/api/decision-logs/` | Returns last 100 decision logs. Optional query: `?model_type=classical|quantum` |
| GET | `/api/simulation-status/` | Returns running state, current session id, weather, quantum flag. |
| GET | `/api/comparison-data/` | Returns aggregated classical vs quantum metrics + chart series. |
| GET | `/api/analytics-data/` | Returns high-level analytics payload for dashboard charts and table. |
| GET | `/api/export-csv/` | Downloads last 100 `VehicleState` rows as CSV (`quantumdrive_logs.csv`). |

### Example Requests

```bash
# Start simulation
curl -X POST http://127.0.0.1:8000/api/start-simulation/ \
    -H "Content-Type: application/json" \
    -d '{}'

# Change weather
curl -X POST http://127.0.0.1:8000/api/change-weather/ \
    -H "Content-Type: application/json" \
    -d '{"weather":"Rain"}'

# Toggle quantum mode
curl -X POST http://127.0.0.1:8000/api/toggle-quantum/ \
    -H "Content-Type: application/json" \
    -d '{"enabled":true}'
```

---

## Data Model Summary

Core entities in `core/models.py`:

- **SimulationSession**: session lifecycle (`running`/`stopped`), start/stop timestamps, quantum flag.
- **VehicleState**: telemetry snapshots + selected model decision/confidence.
- **DecisionLog**: classical/quantum decision events and quality metrics.
- **WeatherCondition**: current weather preset state.

---

## Frontend Behavior & Settings

Frontend app logic lives in `quantumdrive/static/js/app.js` and includes:

- page-wise initializers (dashboard, simulation, logs, comparison, analytics),
- top-bar status synchronization,
- periodic polling for telemetry and logs,
- canvas-based live driving visualization,
- chart rendering and analytics views.

Runtime settings are stored in browser `localStorage` under `qd_*` keys (poll interval, chart points, UI toggles, etc.).

---

## Troubleshooting

- **`python: command not found`**: use `python3` and ensure virtualenv is activated.
- **Port 8000 already in use**: stop existing process, then rerun `runserver`.
- **No live data**: ensure simulation is started from the UI/API.
- **CARLA unavailable**: verify CARLA server availability/config; fallback mode may still provide simulated behavior.

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit changes with clear messages.
4. Open a Pull Request with a concise summary and test notes.

---

## License

Currently intended for research/academic use. Add a `LICENSE` file to define distribution terms explicitly.

---

Built with Django, DRF, Chart.js, and CARLA-driven autonomous driving research concepts.
