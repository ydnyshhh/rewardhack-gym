from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from rewardhack_gym.experiments.metrics import compute_candidate_metrics, grouped_metrics
from rewardhack_gym.experiments.plotting import write_plots
from rewardhack_gym.experiments.reporting import generate_report
from rewardhack_gym.experiments.utils import (
    atomic_write_json,
    atomic_write_jsonl,
    git_commit,
    normalize_rollout_record,
    prepare_run_dir,
)
from rewardhack_gym.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exported Prime/Verifiers RewardHack rollout records.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--experiment-name", default="prime_hosted_eval")
    parser.add_argument("--bootstrap-ci", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    run_dir = prepare_run_dir(args.out, overwrite=args.overwrite)
    candidates = [normalize_rollout_record(record).to_dict() for record in read_jsonl(args.input)]
    summary = compute_candidate_metrics(
        candidates,
        bootstrap=args.bootstrap_ci,
        bootstrap_samples=args.bootstrap_samples,
    )
    metrics_by_model = grouped_metrics(
        candidates,
        lambda record: str(record["model_id"]),
        bootstrap=args.bootstrap_ci,
        bootstrap_samples=args.bootstrap_samples,
    )
    metrics_by_profile = grouped_metrics(
        candidates,
        lambda record: str(record["profile"]),
        bootstrap=args.bootstrap_ci,
        bootstrap_samples=args.bootstrap_samples,
    )
    metrics_by_family = grouped_metrics(
        candidates,
        lambda record: str(record["family"]),
        bootstrap=args.bootstrap_ci,
        bootstrap_samples=args.bootstrap_samples,
    )
    splits = sorted({str(record["split"]) for record in candidates})
    metadata = {
        "run_id": Path(args.out).name,
        "experiment_type": args.experiment_name,
        "timestamp": datetime.fromtimestamp(0, timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config_path": None,
        "input": args.input,
        "num_rollouts": len(candidates),
        "num_tasks": len({str(record["task_id"]) for record in candidates}),
        "models": sorted({str(record["model_id"]) for record in candidates}),
        "families": sorted({str(record["family"]) for record in candidates}),
        "profiles": sorted({str(record["profile"]) for record in candidates}),
        "split": splits[0] if len(splits) == 1 else splits,
    }

    atomic_write_json(run_dir / "metadata.json", metadata)
    atomic_write_json(run_dir / "summary.json", summary.to_dict())
    atomic_write_json(run_dir / "metrics_by_model.json", metrics_by_model)
    atomic_write_json(run_dir / "metrics_by_profile.json", metrics_by_profile)
    atomic_write_json(run_dir / "metrics_by_family.json", metrics_by_family)
    atomic_write_jsonl(run_dir / "candidates.jsonl", candidates)
    atomic_write_jsonl(run_dir / "trajectories.jsonl", candidates)
    atomic_write_jsonl(run_dir / "tasks.jsonl", [])
    atomic_write_jsonl(run_dir / "examples" / "false_passes.jsonl", [item for item in candidates if item["outcome_label"] == "false_pass"][:50])
    atomic_write_jsonl(run_dir / "examples" / "true_passes.jsonl", [item for item in candidates if item["outcome_label"] == "true_pass"][:50])
    atomic_write_jsonl(run_dir / "examples" / "false_fails.jsonl", [item for item in candidates if item["outcome_label"] == "false_fail"][:50])
    atomic_write_jsonl(run_dir / "examples" / "disagreements.jsonl", [item for item in candidates if item["official_passed"] != item["oracle_passed"]][:50])
    write_plots(run_dir, candidates)
    generate_report(
        run_dir=run_dir,
        metadata=metadata,
        summary=summary.to_dict(),
        metrics_by_model=metrics_by_model,
        metrics_by_profile=metrics_by_profile,
        metrics_by_family=metrics_by_family,
        false_pass_examples=[item for item in candidates if item["outcome_label"] == "false_pass"][:10],
    )
    print(run_dir)


if __name__ == "__main__":
    main()
