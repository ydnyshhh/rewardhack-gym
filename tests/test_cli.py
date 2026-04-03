from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from rewardhack_gym import create_environment


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


def test_cli_can_export_mech_interp_rows_and_build_pairs() -> None:
    artifact_dir = Path("tests_artifacts")
    artifact_dir.mkdir(exist_ok=True)
    input_path = artifact_dir / "mech_interp_cli_inputs.jsonl"
    traces_path = artifact_dir / "mech_interp_cli_traces.jsonl"
    export_path = artifact_dir / "mech_interp_cli_rows.jsonl"
    pairs_path = artifact_dir / "mech_interp_cli_pairs.jsonl"

    env = create_environment("math/reasoning-validity")
    task = env.sample_task(seed=5)
    input_path.write_text(
        json.dumps({"seed": 5, "output": env.canonical_true_output(task)}) + "\n" +
        json.dumps({"seed": 5, "output": env.canonical_exploit_output(task)}) + "\n",
        encoding="utf-8",
    )
    run_cli_command(
        "evaluate-traces",
        "math/reasoning-validity",
        "--input",
        str(input_path),
        "--output",
        str(traces_path),
    )
    run_cli_command("export-mech-interp", "--input", str(traces_path), "--output", str(export_path))
    run_cli_command("build-matched-pairs", "--input", str(export_path), "--output", str(pairs_path))

    exported_rows = [json.loads(line) for line in export_path.read_text(encoding="utf-8").strip().splitlines()]
    pair_rows = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(exported_rows) == 2
    assert exported_rows[0]["environment_name"] == "math/reasoning-validity"
    assert "scenario_cohort_id" in exported_rows[0]
    assert len(pair_rows) == 1
    assert pair_rows[0]["match_level"] == "exact-task"
