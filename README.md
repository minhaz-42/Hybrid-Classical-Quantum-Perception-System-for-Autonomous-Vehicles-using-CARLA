# QuantumDrive

Hybrid Classical–Quantum Perception System for Lane, Route, and Traffic-Sign Understanding in Autonomous Vehicles using CARLA.

QuantumDrive is a **production-grade AI research platform** that combines:
- **Knowledge Distillation** — Teacher→Student model compression (ResNet50→MobileNet, ConvNeXt→TinyViT)
- **Quantum-Enhanced ML** — Variational Quantum Circuits (VQC) for autonomous driving decisions via PennyLane
- **Synthetic Data Generation** — Procedural CARLA-style scene generation with domain randomization
- **Experiment Tracking** — Full lifecycle logging with training curves, confusion matrices, ROC analysis
- **Real-Time Simulation** — Live driving visualization with weather/quantum toggling
- **Model Serving & Inference** — REST API for running predictions on trained models
- **Dark Futuristic UI** — Tesla/Waymo-inspired glassmorphism dashboard with neon accents

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
- **Research Lab:** Distillation training curves, confusion matrices, ROC analysis, model architecture comparison.
- **Knowledge Distillation:** Teacher→Student model compression with mixed-precision training, TensorBoard logging.
- **Quantum ML:** Hybrid classical–quantum networks with VQC circuits (PennyLane) and numpy-based fallback simulator.
- **Synthetic Data:** Procedural scene generation with COCO/YOLO export, 7-type domain randomization pipeline.
- **Experiment Tracking:** Create, log, compare, and analyze ML experiments with full metric history.
- **Model Serving API:** `/api/predict/`, `/api/upload-frame/`, `/api/run-model/` for real-time inference.
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
                +--> Classical Service ──> Rule-based decisions
                +--> Quantum Service ───> VQC-enhanced decisions
                +--> Perception Service -> Sensor fusion
                +--> CARLA Service ─────> Simulation (or fallback)
                |
                +--> ML Training Pipeline
                |    ├── Teacher Models (ResNet50, ConvNeXt, ViT)
                |    ├── Student Models (MobileNet, EfficientNet, TinyViT)
                |    └── Distillation (KL + CE + Feature matching)
                |
                +--> Quantum ML Pipeline
                |    ├── PennyLane VQC (real/simulated)
                |    └── Hybrid CNN→Quantum→Decision heads
                |
                +--> Synthetic Data Generation
                |    ├── Procedural scene rendering
                |    └── Domain randomization (7 augmentations)
                |
                +--> Experiment Tracking & Model Serving
                     ├── Experiment lifecycle management
                     └── Inference API (/api/predict, /api/upload-frame)
```

---

## Tech Stack

- **Backend:** Python 3.10+, Django 5.x, Django REST Framework
- **ML/DL:** PyTorch 2.x, torchvision, TensorBoard
- **Quantum:** PennyLane (optional), numpy-based VQC fallback simulator
- **Frontend:** Django Templates, JavaScript (ES6+), Chart.js, Dark Glassmorphism UI
- **Database:** SQLite (default; Postgres-ready)
- **Simulation:** CARLA integration through service layer (procedural fallback)
- **Data Formats:** COCO, YOLO, native JSON

---

## Repository Structure

```text
.
├── manage.py                    # Root wrapper (delegates to quantumdrive/manage.py)
├── requirements.txt
├── README.md
└── quantumdrive/
    ├── manage.py                # Django manage entrypoint
    ├── db.sqlite3
    │
    ├── core/                    # ── Django application ──
    │   ├── models.py            #   ORM models (Simulation, Vehicle, Experiment, …)
    │   ├── serializers.py       #   DRF serializers
    │   ├── urls.py              #   URL routing (pages + API)
    │   ├── views.py             #   Page views & REST endpoints
    │   ├── admin.py
    │   └── services/            #   Service-layer singletons
    │       ├── carla_service.py
    │       ├── perception_service.py
    │       ├── classical_service.py
    │       └── quantum_service.py
    │
    ├── ml_training/             # ── Knowledge Distillation Pipeline ──
    │   ├── dataset/
    │   │   └── driving_dataset.py      # PyTorch Dataset for driving frames
    │   ├── models/
    │   │   ├── teacher_model.py        # ResNet-50 teacher network
    │   │   └── student_model.py        # Lightweight MobileNet student
    │   ├── distillation/
    │   │   └── distillation_loss.py    # KD + feature-map + attention losses
    │   ├── training/
    │   │   └── train_distill.py        # Training loop (AMP, grad accum, TensorBoard)
    │   ├── evaluation/
    │   │   └── evaluate_model.py       # Accuracy, F1, confusion matrix, ROC
    │   └── utils/
    │       └── helpers.py              # Checkpoint I/O, seed, logging
    │
    ├── synthetic_data/          # ── Synthetic Data Generation ──
    │   ├── carla_generator.py          # CARLA scene & sensor orchestrator
    │   ├── randomizer.py               # Domain-randomization (weather, traffic, …)
    │   ├── dataset_writer.py           # Exports COCO / YOLO / raw formats
    │   └── dataset_loader.py           # PyTorch-ready loader with augmentations
    │
    ├── quantum_ml/              # ── Quantum Machine-Learning ──
    │   ├── qnn_model.py                # PennyLane VQC (angle-embed → entangling)
    │   ├── hybrid_network.py           # Classical encoder + quantum layer + head
    │   ├── train_quantum.py            # Training loop for hybrid model
    │   └── quantum_dataset.py          # Feature extraction for quantum input
    │
    ├── experiments/             # ── Experiment Tracking ──
    │   ├── experiment_manager.py       # Create / log / query experiments
    │   └── run_experiment.py           # CLI harness for distillation experiments
    │
    ├── inference/               # ── Model Serving ──
    │   └── inference_service.py        # Load checkpoint → predict (CPU/GPU)
    │
    ├── quantumdrive/            # ── Django project config ──
    │   ├── settings.py
    │   ├── urls.py
    │   ├── asgi.py
    │   └── wsgi.py
    │
    ├── templates/               # ── Jinja2 / Django templates ──
    │   ├── base.html
    │   └── core/
    │       ├── dashboard.html
    │       ├── simulation.html
    │       ├── analytics.html
    │       ├── comparison.html
    │       ├── research_dashboard.html  # Research Lab (charts, inference)
    │       └── …
    │
    └── static/                  # ── Frontend assets ──
        ├── css/style.css               # Dark glassmorphism theme
        └── js/app.js                   # Chart.js, polling, driving viz
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
| **Research & ML** | | |
| GET/POST | `/api/experiments/` | List all experiments or create a new one. |
| GET | `/api/experiments/<id>/` | Experiment detail with nested results & metrics. |
| GET/POST | `/api/experiment-results/` | Query results (`?experiment=<id>`) or post new result. |
| GET | `/api/trained-models/` | List trained model checkpoints. |
| GET/POST | `/api/research-metrics/` | Query metrics (`?experiment=<id>&metric_name=...`) or post new metric. |
| GET | `/api/research-dashboard-data/` | Aggregated dashboard payload (counts, averages, recent experiments). |
| POST | `/api/predict/` | Run inference. Body: `{ "features": [...], "model_type": "student" }` |
| POST | `/api/run-model/` | Run a saved model by ID. Body: `{ "model_id": <id> }` |
| POST | `/api/upload-frame/` | Upload a driving frame (multipart) for perception analysis. |

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
- **Experiment**: research experiment with name, config JSON, model type, status, timestamps.
- **ExperimentResult**: per-epoch metrics (loss, accuracy, val_loss, val_accuracy) linked to an experiment.
- **TrainedModel**: serialized model checkpoint metadata (name, type, path, accuracy, size, creation date).
- **ResearchMetric**: generic metric store (experiment → metric_name → value + timestamp).

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

Built with Django, DRF, PyTorch, PennyLane, Chart.js, and CARLA-driven autonomous driving research concepts.
