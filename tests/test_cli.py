from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def run_cli_command(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "rewardhack_gym.cli", *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_inspect_env_smoke() -> None:
    result = run_cli_command("inspect-env", "code/spec-overfit", "--seed", "2")
    payload = json.loads(result.stdout)
    assert payload["environment"] == "code/spec-overfit"
    assert payload["task"]["family"] == "code"


def test_cli_profile_controls_are_exposed() -> None:
    result = run_cli_command("inspect-env", "math/symbolic-mismatch", "--seed", "5", "--profile", "low")
    payload = json.loads(result.stdout)
    assert payload["profile"] == "low"
    assert payload["exploitability"]["domain_awareness"] == 0.75


def test_cli_sample_batch_writes_records() -> None:
    artifact_dir = Path("tests_artifacts")
    artifact_dir.mkdir(exist_ok=True)
    output_path = artifact_dir / "sample_batch.jsonl"
    run_cli_command(
        "sample-batch",
        "code/patch-verification",
        "--count",
        "3",
        "--output",
        str(output_path),
        "--include-canonicals",
    )
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["environment"] == "code/patch-verification"
    assert "canonical_exploit_output" in first


def test_cli_export_false_passes() -> None:
    artifact_dir = Path("tests_artifacts")
    artifact_dir.mkdir(exist_ok=True)
    input_path = artifact_dir / "cli_in.jsonl"
    output_path = artifact_dir / "cli_out.jsonl"
    input_path.write_text(
        json.dumps({"is_false_pass": True, "task": {"family": "code"}}) + "\n" +
        json.dumps({"is_false_pass": False, "task": {"family": "math"}}) + "\n",
        encoding="utf-8",
    )
    run_cli_command("export-false-passes", "--input", str(input_path), "--output", str(output_path))
    content = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1


def test_cli_evaluate_traces_can_write_summary_and_cohorts() -> None:
    artifact_dir = Path("tests_artifacts")
    artifact_dir.mkdir(exist_ok=True)
    input_path = artifact_dir / "eval_in.jsonl"
    output_path = artifact_dir / "eval_out.jsonl"
    summary_path = artifact_dir / "eval_summary.json"
    false_pass_path = artifact_dir / "eval_false_pass.jsonl"
    input_path.write_text(
        json.dumps({"seed": 5, "output": "EXPR: 1\nDOMAIN: all reals"}) + "\n",
        encoding="utf-8",
    )
    run_cli_command(
        "evaluate-traces",
        "math/constraint-sensitive",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--summary-output",
        str(summary_path),
        "--false-pass-output",
        str(false_pass_path),
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    false_pass_lines = false_pass_path.read_text(encoding="utf-8").strip().splitlines()
    assert summary["num_trajectories"] == 1
    assert summary["false_pass_rate"] == 1.0
    assert len(false_pass_lines) == 1
