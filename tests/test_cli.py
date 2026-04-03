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
