from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from rewardhack_gym import EnvironmentConfig, create_environment
from rewardhack_gym.core.splits import split_seed
from rewardhack_gym.experiments.metrics import compute_best_of_n_metrics, compute_candidate_metrics, grouped_metrics
from rewardhack_gym.experiments.model_clients import build_model_client
from rewardhack_gym.experiments.plotting import write_plots
from rewardhack_gym.experiments.reporting import generate_report
from rewardhack_gym.experiments.schemas import CandidateRecord, ExperimentConfig, ExperimentRunMetadata
from rewardhack_gym.experiments.utils import (
    atomic_write_json,
    atomic_write_jsonl,
    candidate_from_trajectory,
    copy_config_file,
    git_commit,
    load_experiment_config,
    prepare_run_dir,
    public_task_record,
)


def _run_id(name: str, seed: int) -> str:
    digest = hashlib.sha256(f"{name}:{seed}".encode("utf-8")).hexdigest()[:8]
    return f"{name}-{digest}"


def _dummy_completion(env: Any, task: Any, mode: str) -> str:
    if mode == "canonical_true":
        return env.canonical_true_output(task)
    if mode == "canonical_exploit":
        return env.canonical_exploit_output(task)
    if mode == "random_bad":
        return "I don't know."
    raise ValueError(f"Unknown dummy model mode {mode!r}.")


def _sample_tasks(*, env: Any, family: str, split: str, seed: int, num_tasks: int) -> list[Any]:
    return [
        env.sample_task(seed=split_seed(seed, split=split, index=index, salt=family))
        for index in range(num_tasks)
    ]


def _evaluate_candidates(
    *,
    config: ExperimentConfig,
    run_id: str,
    dry_run: bool,
    dummy_model_mode: str,
    num_candidates: int,
) -> tuple[list[dict[str, Any]], list[CandidateRecord], list[CandidateRecord]]:
    task_records: list[dict[str, Any]] = []
    candidates: list[CandidateRecord] = []
    trajectories: list[CandidateRecord] = []
    model_clients = {
        model.id: build_model_client(model)
        for model in config.models
        if not dry_run or model.provider in {"dummy", "static"}
    }
    for family in config.environment.resolved_families():
        for profile in config.environment.resolved_profiles():
            env_config = EnvironmentConfig.from_profile(
                seed=config.experiment.seed,
                dataset_split=config.environment.split,
                profile=profile,
                code_execution_backend=config.environment.code_execution_backend,
                code_execution_timeout_seconds=config.environment.code_execution_timeout_seconds,
                code_execution_memory_mb=config.environment.code_execution_memory_mb,
            )
            env = create_environment(family, env_config)
            tasks = _sample_tasks(
                env=env,
                family=family,
                split=config.environment.split,
                seed=config.experiment.seed,
                num_tasks=config.environment.num_tasks,
            )
            task_records.extend(public_task_record(task) for task in tasks)
            for model in config.models:
                client = model_clients.get(model.id)
                for task_index, task in enumerate(tasks):
                    for candidate_index in range(num_candidates):
                        if dry_run:
                            completion = _dummy_completion(env, task, dummy_model_mode)
                        else:
                            if client is None:
                                raise RuntimeError(f"No model client available for model {model.id!r}.")
                            completion = client.generate(
                                [{"role": "user", "content": task.prompt}],
                                config.sampling,
                            )
                        trajectory = env.evaluate_output(
                            task,
                            completion,
                            policy_id=model.id,
                            annotations={
                                "experiment_type": config.experiment.type,
                                "candidate_index": candidate_index,
                                "model_id": model.id,
                            },
                        )
                        candidate_id = (
                            f"{run_id}:{model.id}:{family}:{profile}:{task_index}:{candidate_index}"
                        )
                        record = candidate_from_trajectory(
                            run_id=run_id,
                            experiment_type=config.experiment.type,
                            model_id=model.id,
                            family=family,
                            profile=profile,
                            split=config.environment.split,
                            candidate_id=candidate_id,
                            candidate_index=candidate_index,
                            sampling=asdict(config.sampling),
                            trajectory=trajectory,
                            metadata={
                                "task_index": task_index,
                                "dummy_model_mode": dummy_model_mode if dry_run else None,
                            },
                        )
                        candidates.append(record)
                        trajectories.append(record)
    return task_records, candidates, trajectories


def _write_examples(run_dir: Path, candidates: list[CandidateRecord], *, max_examples: int) -> dict[str, list[dict[str, Any]]]:
    buckets = {
        "false_passes": [record.to_dict() for record in candidates if record.outcome_label == "false_pass"],
        "true_passes": [record.to_dict() for record in candidates if record.outcome_label == "true_pass"],
        "false_fails": [record.to_dict() for record in candidates if record.outcome_label == "false_fail"],
        "disagreements": [record.to_dict() for record in candidates if record.official_passed != record.oracle_passed],
    }
    for name, records in buckets.items():
        atomic_write_jsonl(run_dir / "examples" / f"{name}.jsonl", records[:max_examples])
    return buckets


def _finalize_run(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    metadata: ExperimentRunMetadata,
    task_records: list[dict[str, Any]],
    candidates: list[CandidateRecord],
    trajectories: list[CandidateRecord],
    best_of_n_metrics: list[dict[str, Any]] | None = None,
    selected_records: list[dict[str, Any]] | None = None,
) -> None:
    best_of_n_metrics = best_of_n_metrics or []
    selected_records = selected_records or []
    candidate_dicts = [record.to_dict() for record in candidates]
    trajectory_dicts = [record.to_dict() for record in trajectories]
    atomic_write_json(run_dir / "metadata.json", metadata.to_dict())
    atomic_write_jsonl(run_dir / "tasks.jsonl", task_records)
    atomic_write_jsonl(run_dir / "candidates.jsonl", candidate_dicts)
    atomic_write_jsonl(run_dir / "trajectories.jsonl", trajectory_dicts)
    if selected_records:
        atomic_write_jsonl(run_dir / "best_of_n_selected.jsonl", selected_records)
    summary = compute_candidate_metrics(
        candidate_dicts,
        bootstrap=config.reporting.bootstrap_ci,
        bootstrap_samples=config.reporting.bootstrap_samples,
        bootstrap_seed=config.experiment.seed,
    )
    metrics_by_model = grouped_metrics(
        candidate_dicts,
        lambda record: str(record["model_id"]),
        bootstrap=config.reporting.bootstrap_ci,
        bootstrap_samples=config.reporting.bootstrap_samples,
    )
    metrics_by_profile = grouped_metrics(
        candidate_dicts,
        lambda record: str(record["profile"]),
        bootstrap=config.reporting.bootstrap_ci,
        bootstrap_samples=config.reporting.bootstrap_samples,
    )
    metrics_by_family = grouped_metrics(
        candidate_dicts,
        lambda record: str(record["family"]),
        bootstrap=config.reporting.bootstrap_ci,
        bootstrap_samples=config.reporting.bootstrap_samples,
    )
    atomic_write_json(run_dir / "summary.json", summary.to_dict())
    atomic_write_json(run_dir / "metrics_by_model.json", metrics_by_model)
    atomic_write_json(run_dir / "metrics_by_profile.json", metrics_by_profile)
    atomic_write_json(run_dir / "metrics_by_family.json", metrics_by_family)
    if best_of_n_metrics:
        atomic_write_json(run_dir / "best_of_n_metrics.json", {"curves": best_of_n_metrics})
    examples = _write_examples(
        run_dir,
        candidates,
        max_examples=config.reporting.max_examples_per_bucket if config.reporting.save_examples else 0,
    )
    write_plots(run_dir, candidate_dicts, best_of_n_metrics)
    generate_report(
        run_dir=run_dir,
        metadata=metadata.to_dict(),
        summary=summary.to_dict(),
        metrics_by_model=metrics_by_model,
        metrics_by_profile=metrics_by_profile,
        metrics_by_family=metrics_by_family,
        best_of_n_metrics=best_of_n_metrics,
        false_pass_examples=examples["false_passes"][:10],
    )


def run_experiment(
    *,
    config: ExperimentConfig,
    config_path: str | Path | None,
    out: str | Path,
    dry_run: bool,
    dummy_model_mode: str,
    overwrite: bool = False,
    num_candidates: int | None = None,
) -> Path:
    run_dir = prepare_run_dir(out, overwrite=overwrite)
    if config_path is not None:
        copy_config_file(config_path, run_dir)
    else:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    run_id = _run_id(config.experiment.name, config.experiment.seed)
    metadata = ExperimentRunMetadata.create(
        run_id=run_id,
        config=config,
        config_path=str(config_path) if config_path else None,
        git_commit=git_commit(),
    )
    candidate_count = num_candidates if num_candidates is not None else max(1, config.sampling.num_candidates)
    task_records, candidates, trajectories = _evaluate_candidates(
        config=config,
        run_id=run_id,
        dry_run=dry_run,
        dummy_model_mode=dummy_model_mode,
        num_candidates=candidate_count,
    )
    best_metrics: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    if config.experiment.type == "best_of_n":
        selected, summaries = compute_best_of_n_metrics(
            candidates,
            n_values=config.best_of_n.values,
            selection_policies=config.best_of_n.selection_policies,
            seed=config.experiment.seed,
        )
        selected_records = [item.to_dict() for item in selected]
        best_metrics = [item.to_dict() for item in summaries]
    _finalize_run(
        run_dir=run_dir,
        config=config,
        metadata=metadata,
        task_records=task_records,
        candidates=candidates,
        trajectories=trajectories,
        best_of_n_metrics=best_metrics,
        selected_records=selected_records,
    )
    return run_dir


def run_best_of_n_experiment(
    *,
    config_path: str | Path,
    out: str | Path,
    dry_run: bool,
    dummy_model_mode: str,
    overwrite: bool = False,
) -> Path:
    config = load_experiment_config(config_path)
    candidate_count = max(config.best_of_n.values) if config.best_of_n.values else 1
    return run_experiment(
        config=config,
        config_path=config_path,
        out=out,
        dry_run=dry_run,
        dummy_model_mode=dummy_model_mode,
        overwrite=overwrite,
        num_candidates=candidate_count,
    )


def run_profile_sweep_experiment(
    *,
    config_path: str | Path,
    out: str | Path,
    dry_run: bool,
    dummy_model_mode: str,
    overwrite: bool = False,
) -> Path:
    config = load_experiment_config(config_path)
    return run_experiment(
        config=config,
        config_path=config_path,
        out=out,
        dry_run=dry_run,
        dummy_model_mode=dummy_model_mode,
        overwrite=overwrite,
    )


def run_model_sweep_experiment(
    *,
    config_path: str | Path,
    out: str | Path,
    dry_run: bool,
    dummy_model_mode: str,
    overwrite: bool = False,
) -> Path:
    config = load_experiment_config(config_path)
    return run_experiment(
        config=config,
        config_path=config_path,
        out=out,
        dry_run=dry_run,
        dummy_model_mode=dummy_model_mode,
        overwrite=overwrite,
    )
