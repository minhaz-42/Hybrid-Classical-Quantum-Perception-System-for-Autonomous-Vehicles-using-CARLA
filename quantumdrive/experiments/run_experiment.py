"""
QuantumDrive — Run Experiment
===============================
CLI script to run a complete ML experiment with tracking.

Usage:
    python -m experiments.run_experiment \\
        --name "ResNet50→MobileNet Distillation" \\
        --type distillation \\
        --teacher resnet50 \\
        --student mobilenet \\
        --epochs 50
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.experiment_manager import experiment_manager


def run_distillation_experiment(config: Dict[str, Any]):
    """Run a knowledge distillation experiment."""
    from ml_training.training.train_distill import DistillationTrainer

    exp_id = experiment_manager.create_experiment(
        name=config["name"],
        model_name=f"{config['teacher']}→{config['student']}",
        dataset_name=config.get("dataset", "synthetic"),
        config=config,
        tags=["distillation", config["teacher"], config["student"]],
    )

    try:
        trainer = DistillationTrainer(config)
        trainer.train()

        # Log evaluation metrics
        experiment_manager.log_evaluation(exp_id, {
            "best_val_loss": trainer.best_val_loss,
            "final_epoch": len(trainer.history["train_loss"]),
            "history": trainer.history,
        })

        experiment_manager.add_artifact(
            exp_id,
            str(trainer.output_dir / "checkpoint_best.pt"),
            "checkpoint",
        )
        experiment_manager.finish_experiment(exp_id, "completed")

    except Exception as e:
        experiment_manager.finish_experiment(exp_id, f"failed: {str(e)}")
        raise

    return exp_id


def run_quantum_experiment(config: Dict[str, Any]):
    """Run a quantum ML experiment."""
    from quantum_ml.train_quantum import QuantumTrainer

    exp_id = experiment_manager.create_experiment(
        name=config["name"],
        model_name=f"HybridQNN-{config['n_qubits']}q-{config['n_layers']}l",
        dataset_name="quantum_synthetic",
        config=config,
        tags=["quantum", f"{config['n_qubits']}qubits"],
    )

    try:
        trainer = QuantumTrainer(config)
        trainer.train()

        experiment_manager.log_evaluation(exp_id, {
            "history": trainer.history,
        })
        experiment_manager.finish_experiment(exp_id, "completed")

    except Exception as e:
        experiment_manager.finish_experiment(exp_id, f"failed: {str(e)}")
        raise

    return exp_id


def main():
    parser = argparse.ArgumentParser(description="QuantumDrive — Run Experiment")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--type", type=str, default="distillation",
                        choices=["distillation", "quantum", "evaluation"])
    # Distillation args
    parser.add_argument("--teacher", type=str, default="resnet50")
    parser.add_argument("--student", type=str, default="mobilenet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    # Quantum args
    parser.add_argument("--n_qubits", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)

    args = parser.parse_args()
    config = vars(args)
    exp_type = config.pop("type")

    if exp_type == "distillation":
        exp_id = run_distillation_experiment(config)
    elif exp_type == "quantum":
        exp_id = run_quantum_experiment(config)
    else:
        print(f"Unknown experiment type: {exp_type}")
        sys.exit(1)

    print(f"\n[QuantumDrive] Experiment completed: {exp_id}")

    # Print summary
    exp = experiment_manager.get_experiment(exp_id)
    if exp:
        print(f"  Name: {exp['name']}")
        print(f"  Status: {exp['status']}")
        print(f"  Runtime: {exp.get('runtime_seconds', 0):.1f}s")


if __name__ == "__main__":
    main()
