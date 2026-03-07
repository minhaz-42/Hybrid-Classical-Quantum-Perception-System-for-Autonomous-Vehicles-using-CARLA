"""
QuantumDrive — Experiment Manager
====================================
Manages ML experiments lifecycle: creation, tracking, comparison, and persistence.

Provides a unified interface for:
    - Registering new experiments
    - Logging metrics during training
    - Comparing multiple experiment runs
    - Persisting results to Django database
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib
import platform


class ExperimentManager:
    """
    Manages experiment lifecycle and metric tracking.

    Each experiment records:
        - Model architecture and hyperparameters
        - Dataset information
        - Training metrics (loss, accuracy per epoch)
        - Evaluation metrics
        - Hardware configuration
        - Runtime duration
    """

    def __init__(self, storage_dir: str = "./experiment_results"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.active_experiments: Dict[str, Dict] = {}

    def create_experiment(
        self,
        name: str,
        model_name: str,
        dataset_name: str,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Register a new experiment.

        Returns:
            Unique experiment ID
        """
        timestamp = datetime.now().isoformat()
        exp_id = hashlib.md5(f"{name}_{timestamp}".encode()).hexdigest()[:12]

        experiment = {
            "id": exp_id,
            "name": name,
            "model": model_name,
            "dataset": dataset_name,
            "config": config,
            "tags": tags or [],
            "status": "running",
            "created_at": timestamp,
            "updated_at": timestamp,
            "hardware": self._get_hardware_info(),
            "metrics": {
                "train_loss": [],
                "val_loss": [],
                "val_accuracy": [],
                "learning_rate": [],
            },
            "evaluation": {},
            "artifacts": [],
            "runtime_seconds": 0,
            "start_time": time.time(),
        }

        self.active_experiments[exp_id] = experiment
        print(f"[Experiment] Created: {name} (ID: {exp_id})")
        return exp_id

    def log_metrics(self, exp_id: str, step: int, **metrics):
        """Log training metrics for a given step."""
        exp = self.active_experiments.get(exp_id)
        if not exp:
            raise ValueError(f"Experiment {exp_id} not found")

        for key, value in metrics.items():
            if key not in exp["metrics"]:
                exp["metrics"][key] = []
            exp["metrics"][key].append({"step": step, "value": float(value)})

        exp["updated_at"] = datetime.now().isoformat()

    def log_evaluation(self, exp_id: str, eval_results: Dict[str, Any]):
        """Log final evaluation results."""
        exp = self.active_experiments.get(exp_id)
        if not exp:
            raise ValueError(f"Experiment {exp_id} not found")
        exp["evaluation"] = eval_results
        exp["updated_at"] = datetime.now().isoformat()

    def add_artifact(self, exp_id: str, artifact_path: str, artifact_type: str = "checkpoint"):
        """Register an artifact (checkpoint, plot, etc.)."""
        exp = self.active_experiments.get(exp_id)
        if not exp:
            raise ValueError(f"Experiment {exp_id} not found")
        exp["artifacts"].append({
            "path": artifact_path,
            "type": artifact_type,
            "timestamp": datetime.now().isoformat(),
        })

    def finish_experiment(self, exp_id: str, status: str = "completed"):
        """Mark experiment as completed and save."""
        exp = self.active_experiments.get(exp_id)
        if not exp:
            raise ValueError(f"Experiment {exp_id} not found")

        exp["status"] = status
        exp["runtime_seconds"] = time.time() - exp.pop("start_time", time.time())
        exp["updated_at"] = datetime.now().isoformat()

        # Save to disk
        output_path = self.storage_dir / f"{exp_id}.json"
        with open(output_path, "w") as f:
            json.dump(exp, f, indent=2)

        print(f"[Experiment] Finished: {exp['name']} ({status}) — {exp['runtime_seconds']:.1f}s")
        return exp

    def compare_experiments(self, exp_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics across multiple experiments."""
        experiments = []
        for eid in exp_ids:
            exp = self.active_experiments.get(eid)
            if not exp:
                # Try loading from disk
                path = self.storage_dir / f"{eid}.json"
                if path.exists():
                    with open(path) as f:
                        exp = json.load(f)
            if exp:
                experiments.append(exp)

        if not experiments:
            return {"error": "No experiments found"}

        comparison = {
            "experiments": [],
            "best_accuracy": {"id": None, "value": 0},
            "best_loss": {"id": None, "value": float("inf")},
        }

        for exp in experiments:
            val_acc = exp.get("evaluation", {}).get("accuracy", 0)
            val_loss = exp.get("evaluation", {}).get("loss", float("inf"))

            summary = {
                "id": exp["id"],
                "name": exp["name"],
                "model": exp["model"],
                "status": exp["status"],
                "accuracy": val_acc,
                "loss": val_loss,
                "runtime": exp.get("runtime_seconds", 0),
                "config": exp.get("config", {}),
            }
            comparison["experiments"].append(summary)

            if val_acc > comparison["best_accuracy"]["value"]:
                comparison["best_accuracy"] = {"id": exp["id"], "value": val_acc}
            if val_loss < comparison["best_loss"]["value"]:
                comparison["best_loss"] = {"id": exp["id"], "value": val_loss}

        return comparison

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments (active + stored)."""
        experiments = list(self.active_experiments.values())

        # Load stored experiments
        for path in self.storage_dir.glob("*.json"):
            try:
                with open(path) as f:
                    exp = json.load(f)
                if exp.get("id") not in self.active_experiments:
                    experiments.append(exp)
            except (json.JSONDecodeError, KeyError):
                continue

        return sorted(experiments, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Get a single experiment by ID."""
        if exp_id in self.active_experiments:
            return self.active_experiments[exp_id]

        path = self.storage_dir / f"{exp_id}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    @staticmethod
    def _get_hardware_info() -> Dict[str, Any]:
        """Collect hardware information."""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }
        try:
            import torch
            info["pytorch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_mem / 1e9
        except ImportError:
            info["pytorch_version"] = "not installed"
        return info


# Module-level singleton
experiment_manager = ExperimentManager()
