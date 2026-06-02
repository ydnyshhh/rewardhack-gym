from __future__ import annotations

import json
from pathlib import Path

import yaml

from rewardhack_gym.experiments.runners import run_best_of_n_experiment
from rewardhack_gym.io import read_jsonl


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


def test_dummy_best_of_n_runner_writes_expected_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    run_dir = tmp_path / "run"
    write_config(config_path)

    run_best_of_n_experiment(
        config_path=config_path,
        out=run_dir,
        dry_run=True,
        dummy_model_mode="canonical_exploit",
    )

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "tasks.jsonl").exists()
    assert (run_dir / "candidates.jsonl").exists()
    assert (run_dir / "trajectories.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "examples" / "false_passes.jsonl").exists()
    assert (run_dir / "report.md").exists()
    assert list((run_dir / "plots").glob("*.png"))

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["false_pass_rate"] > 0
    assert read_jsonl(run_dir / "candidates.jsonl")

