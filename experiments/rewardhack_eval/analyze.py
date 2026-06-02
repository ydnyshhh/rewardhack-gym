from __future__ import annotations

import argparse
import json
from pathlib import Path

from rewardhack_gym.experiments.metrics import compute_candidate_metrics, grouped_metrics
from rewardhack_gym.experiments.utils import atomic_write_json
from rewardhack_gym.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute RewardHack-Gym experiment summaries.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    candidates = read_jsonl(run_dir / "candidates.jsonl")
    summary = compute_candidate_metrics(candidates, bootstrap=True)
    atomic_write_json(run_dir / "summary.json", summary.to_dict())
    atomic_write_json(run_dir / "metrics_by_model.json", grouped_metrics(candidates, lambda item: str(item["model_id"])))
    atomic_write_json(run_dir / "metrics_by_profile.json", grouped_metrics(candidates, lambda item: str(item["profile"])))
    atomic_write_json(run_dir / "metrics_by_family.json", grouped_metrics(candidates, lambda item: str(item["family"])))
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

