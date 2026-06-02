from __future__ import annotations

from pathlib import Path

import yaml

from rewardhack_gym.experiments.runners import run_best_of_n_experiment


FORBIDDEN_PUBLIC_STRINGS = (
    "hidden_cases",
    "oracle_property_cases",
    "canonical_exploit_output",
    "canonical_true_output",
    "hidden_metadata",
)


def write_config(path: Path) -> None:
    payload = {
        "experiment": {"name": "tiny_best_of_n", "type": "best_of_n", "seed": 0},
        "environment": {
            "family": "code/spec-overfit",
            "profiles": ["medium"],
            "split": "eval",
            "num_tasks": 2,
            "code_execution_backend": "subprocess",
            "code_execution_timeout_seconds": 2.0,
            "code_execution_memory_mb": 256,
        },
        "models": [{"id": "dummy", "provider": "dummy"}],
        "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024},
        "best_of_n": {"values": [1, 2], "selection_policies": ["official_only"]},
        "reporting": {"save_examples": True, "max_examples_per_bucket": 10, "bootstrap_ci": False},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_experiment_public_outputs_do_not_leak_hidden_or_canonical_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    run_dir = tmp_path / "run"
    write_config(config_path)

    run_best_of_n_experiment(
        config_path=config_path,
        out=run_dir,
        dry_run=True,
        dummy_model_mode="canonical_exploit",
    )

    public_paths = [
        run_dir / "tasks.jsonl",
        run_dir / "candidates.jsonl",
        run_dir / "trajectories.jsonl",
        run_dir / "report.md",
        run_dir / "examples" / "false_passes.jsonl",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in public_paths if path.exists())
    for forbidden in FORBIDDEN_PUBLIC_STRINGS:
        assert forbidden not in combined
