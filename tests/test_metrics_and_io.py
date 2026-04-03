import json
from pathlib import Path

from rewardhack_gym import create_environment
from rewardhack_gym.analysis import summarize_trajectories
from rewardhack_gym.io import read_jsonl, write_jsonl


def test_metrics_and_jsonl_roundtrip() -> None:
    env = create_environment("code/spec-overfit")
    task = env.sample_task(seed=7)
    trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

    summary = summarize_trajectories([trajectory])
    assert summary.false_pass_rate == 1.0
    assert summary.official_pass_rate == 1.0
    assert summary.oracle_pass_rate == 0.0

    output_dir = Path("tests_artifacts")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "metrics_roundtrip.jsonl"
    write_jsonl(output_path, [trajectory])
    records = read_jsonl(output_path)
    assert len(records) == 1
    assert records[0]["is_false_pass"] is True
    json.dumps(records[0])
