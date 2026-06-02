from __future__ import annotations

import argparse
import json
from pathlib import Path

from rewardhack_gym.experiments.reporting import generate_report
from rewardhack_gym.io import read_jsonl


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a RewardHack-Gym experiment report.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--format", choices=("markdown",), default="markdown")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    false_pass_path = run_dir / "examples" / "false_passes.jsonl"
    bon_path = run_dir / "best_of_n_metrics.json"
    bon = _read_json(bon_path).get("curves", []) if bon_path.exists() else []
    report = generate_report(
        run_dir=run_dir,
        metadata=_read_json(run_dir / "metadata.json"),
        summary=_read_json(run_dir / "summary.json"),
        metrics_by_model=_read_json(run_dir / "metrics_by_model.json"),
        metrics_by_profile=_read_json(run_dir / "metrics_by_profile.json"),
        metrics_by_family=_read_json(run_dir / "metrics_by_family.json"),
        best_of_n_metrics=bon,
        false_pass_examples=read_jsonl(false_pass_path) if false_pass_path.exists() else [],
        output_format=args.format,
    )
    print(run_dir / "report.md")


if __name__ == "__main__":
    main()

