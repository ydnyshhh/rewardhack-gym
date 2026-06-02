from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from rewardhack_gym.core.models import JSONValue, Task, Trajectory, serialize_value
from rewardhack_gym.experiments.schemas import CandidateRecord, ExperimentConfig, outcome_label, redact_public_metadata


PRIVATE_METADATA_FRAGMENTS = ("hidden", "oracle")
PRIVATE_METADATA_KEYS = {"canonical_true_output", "canonical_exploit_output"}
SENSITIVE_CONFIG_KEYS = {
    "api_key",
    "authorization",
    "access_token",
    "bearer_token",
    "password",
    "secret",
    "token",
}


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return ExperimentConfig.from_dict(payload)


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True))
            handle.write("\n")
    tmp.replace(target)


def copy_config_file(config_path: str | Path, run_dir: str | Path) -> None:
    target = Path(run_dir) / "config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    target.write_text(yaml.safe_dump(redact_experiment_config(payload), sort_keys=False), encoding="utf-8")


def prepare_run_dir(run_dir: str | Path, *, overwrite: bool) -> Path:
    target = Path(run_dir)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Run directory {target} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(target)
    (target / "examples").mkdir(parents=True, exist_ok=True)
    (target / "plots").mkdir(parents=True, exist_ok=True)
    return target


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _is_private_key(key: object) -> bool:
    normalized = str(key).lower()
    if normalized in PRIVATE_METADATA_KEYS:
        return True
    return any(fragment in normalized for fragment in PRIVATE_METADATA_FRAGMENTS)


def _is_sensitive_config_key(key: object) -> bool:
    normalized = str(key).lower()
    if normalized in SENSITIVE_CONFIG_KEYS:
        return True
    return "header" in normalized


def redact_experiment_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): ("<redacted>" if _is_sensitive_config_key(key) else redact_experiment_config(item))
            for key, item in value.items()
            if "header" not in str(key).lower()
        }
    if isinstance(value, list):
        return [redact_experiment_config(item) for item in value]
    return value


def redact_public_value(value: Any) -> JSONValue:
    serialized = serialize_value(value)
    if isinstance(serialized, dict):
        return {
            str(key): redact_public_value(item)
            for key, item in serialized.items()
            if not _is_private_key(key)
        }
    if isinstance(serialized, list):
        return [redact_public_value(item) for item in serialized]
    return serialized


def public_task_record(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "family": task.family,
        "prompt": task.prompt,
        "expected_interface": task.expected_interface,
        "difficulty": task.difficulty,
        "exploit_surface": task.exploit_surface.to_dict(),
        "metadata": redact_public_value(task.metadata),
        "tags": list(task.tags),
    }


def execution_backend_from_trajectory(trajectory: Trajectory) -> str | None:
    for result in (trajectory.official_result, trajectory.oracle_result):
        for component in result.components:
            execution = component.diagnostics.get("execution")
            if isinstance(execution, Mapping):
                backend = execution.get("backend")
                if backend is not None:
                    return str(backend)
    return None


def semantic_failures_from_trajectory(trajectory: Trajectory) -> list[str]:
    failures = trajectory.annotations.get("semantic_failures", [])
    if isinstance(failures, list):
        return [str(item) for item in failures]
    return []


def candidate_from_trajectory(
    *,
    run_id: str,
    experiment_type: str,
    model_id: str,
    family: str,
    profile: str,
    split: str,
    candidate_id: str,
    candidate_index: int,
    sampling: Mapping[str, Any],
    trajectory: Trajectory,
    metadata: Mapping[str, Any] | None = None,
) -> CandidateRecord:
    official_passed = trajectory.official_result.passed
    oracle_passed = trajectory.oracle_result.passed
    return CandidateRecord(
        run_id=run_id,
        experiment_type=experiment_type,
        model_id=model_id,
        family=family,
        profile=profile,
        split=split,
        task_id=trajectory.task.task_id,
        candidate_id=candidate_id,
        candidate_index=candidate_index,
        sampling=dict(sampling),
        prompt=trajectory.prompt,
        completion=trajectory.final_output,
        official_score=float(trajectory.official_result.score),
        official_passed=official_passed,
        oracle_score=float(trajectory.oracle_result.score),
        oracle_passed=oracle_passed,
        hack_gap=float(trajectory.official_result.score - trajectory.oracle_result.score),
        false_pass=official_passed and not oracle_passed,
        outcome_label=outcome_label(official_passed, oracle_passed),
        exploit_labels=[str(label) for label in trajectory.exploit_labels],
        semantic_failures=semantic_failures_from_trajectory(trajectory),
        execution_backend=execution_backend_from_trajectory(trajectory),
        duration_seconds=trajectory.runtime.duration_seconds,
        metadata=redact_public_metadata(dict(metadata or {})),
    )


def normalize_rollout_record(record: dict[str, Any]) -> CandidateRecord:
    task = record.get("task") or record.get("state", {}).get("task") or {}
    info = task.get("info", {}) if isinstance(task, Mapping) else {}
    task_metadata = {}
    if isinstance(task, Mapping) and isinstance(task.get("metadata"), Mapping):
        task_metadata = dict(task["metadata"])
    if isinstance(info, Mapping) and isinstance(info.get("metadata"), Mapping):
        task_metadata.update(info["metadata"])
    state = record.get("state") or {}
    trajectory = state.get("rewardhack_trajectory") or record.get("rewardhack_trajectory") or {}
    official_score = float(record.get("official_reward", record.get("official_score", trajectory.get("official_result", {}).get("score", 0.0))))
    oracle_score = float(record.get("oracle_score", trajectory.get("oracle_result", {}).get("score", 0.0)))
    false_pass = bool(record.get("false_pass", trajectory.get("is_false_pass", False)))
    official_passed = bool(record.get("official_passed", trajectory.get("official_result", {}).get("passed", official_score > 0.0)))
    oracle_passed = bool(record.get("oracle_passed", trajectory.get("oracle_result", {}).get("passed", oracle_score > 0.0)))
    completion = record.get("completion", state.get("completion", ""))
    if isinstance(completion, list) and completion:
        last = completion[-1]
        completion_text = str(last.get("content", last)) if isinstance(last, Mapping) else str(last)
    else:
        completion_text = str(completion or "")
    prompt_value = task.get("prompt", record.get("prompt", "")) if isinstance(task, Mapping) else record.get("prompt", "")
    if isinstance(prompt_value, list) and prompt_value:
        first_prompt = prompt_value[0]
        prompt_text = str(first_prompt.get("content", first_prompt)) if isinstance(first_prompt, Mapping) else str(first_prompt)
    else:
        prompt_text = str(prompt_value or "")
    task_id = str(task.get("task_id", info.get("task_id", record.get("task_id", "unknown"))))
    family = str(task.get("family", info.get("family", record.get("family", "unknown"))))
    profile = str(record.get("profile", task_metadata.get("profile", "unknown")))
    split = str(record.get("split", task_metadata.get("split", "eval")))
    raw_metadata = dict(record.get("metadata", {})) if isinstance(record.get("metadata", {}), Mapping) else {}
    model = record.get("model", {})
    if isinstance(model, Mapping):
        if model.get("id") is not None:
            raw_metadata.setdefault("model_id", str(model["id"]))
        if model.get("model_path") is not None:
            raw_metadata.setdefault("model_path", str(model["model_path"]))
        if model.get("provider") is not None:
            raw_metadata.setdefault("model_provider", str(model["provider"]))
    if record.get("model_path") is not None:
        raw_metadata.setdefault("model_path", str(record["model_path"]))
    if record.get("model_provider") is not None:
        raw_metadata.setdefault("model_provider", str(record["model_provider"]))
    sampling = record.get("sampling")
    if not isinstance(sampling, Mapping):
        sampling = state.get("sampling") if isinstance(state, Mapping) else {}
    if not isinstance(sampling, Mapping):
        sampling = {}
    return CandidateRecord(
        run_id=str(record.get("run_id", "prime-rollout")),
        experiment_type=str(record.get("experiment_type", "prime_eval")),
        model_id=str(record.get("model_id", record.get("policy_id", "unknown"))),
        family=family,
        profile=profile,
        split=split,
        task_id=task_id,
        candidate_id=str(record.get("candidate_id", task_id)),
        candidate_index=int(record.get("candidate_index", 0)),
        sampling=dict(sampling),
        prompt=prompt_text,
        completion=completion_text,
        official_score=official_score,
        official_passed=official_passed,
        oracle_score=oracle_score,
        oracle_passed=oracle_passed,
        hack_gap=float(record.get("hack_gap", official_score - oracle_score)),
        false_pass=false_pass,
        outcome_label=outcome_label(official_passed, oracle_passed),
        exploit_labels=[str(item) for item in record.get("exploit_labels", trajectory.get("exploit_labels", []))],
        semantic_failures=[str(item) for item in record.get("semantic_failures", trajectory.get("annotations", {}).get("semantic_failures", []))],
        execution_backend=record.get("execution_backend"),
        duration_seconds=record.get("duration_seconds"),
        metadata=raw_metadata,
    )
