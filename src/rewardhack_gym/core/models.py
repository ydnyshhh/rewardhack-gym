from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def serialize_value(value: Any) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, tuple):
        return [serialize_value(item) for item in value]
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    if is_dataclass(value):
        data = {field_.name: serialize_value(getattr(value, field_.name)) for field_ in fields(value)}
        return data
    return str(value)


@dataclass(frozen=True, slots=True)
class ExploitSurface:
    """Describes how the official verifier can be gamed."""

    summary: str
    ignored_dimensions: tuple[str, ...]
    exploit_families: tuple[str, ...]
    cheaper_path_hypotheses: tuple[str, ...]
    control_dimensions: Mapping[str, JSONScalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        return serialize_value(self)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Task:
    """Serializable task object."""

    task_id: str
    family: str
    prompt: str
    expected_interface: str
    difficulty: str
    exploit_surface: ExploitSurface
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    hidden_metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def to_dict(self, *, include_hidden: bool = False) -> dict[str, JSONValue]:
        payload = {
            "task_id": self.task_id,
            "family": self.family,
            "prompt": self.prompt,
            "expected_interface": self.expected_interface,
            "difficulty": self.difficulty,
            "exploit_surface": self.exploit_surface.to_dict(),
            "metadata": serialize_value(self.metadata),
            "tags": serialize_value(self.tags),
        }
        if include_hidden:
            payload["hidden_metadata"] = serialize_value(self.hidden_metadata)
        return payload


@dataclass(frozen=True, slots=True)
class CheckerResult:
    checker_name: str
    score: float
    passed: bool
    diagnostics: Mapping[str, JSONValue] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, JSONValue]:
        return serialize_value(self)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class EvaluatorResult:
    evaluator_name: str
    score: float
    passed: bool
    components: tuple[CheckerResult, ...]
    diagnostics: Mapping[str, JSONValue] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, JSONValue]:
        return serialize_value(self)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RuntimeMetadata:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    policy_id: str | None = None
    rollout_version_id: str | None = None
    evaluator_version_id: str | None = None
    environment_profile: str | None = None
    extra: Mapping[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        return serialize_value(self)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    role: str
    content: str
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        return serialize_value(self)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Trajectory:
    task: Task
    prompt: str
    final_output: str
    official_result: EvaluatorResult
    oracle_result: EvaluatorResult
    exploit_labels: tuple[str, ...]
    runtime: RuntimeMetadata = field(default_factory=RuntimeMetadata)
    steps: tuple[TrajectoryStep, ...] = ()
    annotations: Mapping[str, JSONValue] = field(default_factory=dict)

    @property
    def verifier_gap(self) -> float:
        return self.official_result.score - self.oracle_result.score

    @property
    def is_false_pass(self) -> bool:
        return self.official_result.passed and not self.oracle_result.passed

    def to_dict(self, *, include_hidden_task_metadata: bool = False) -> dict[str, JSONValue]:
        return {
            "task": self.task.to_dict(include_hidden=include_hidden_task_metadata),
            "prompt": self.prompt,
            "final_output": self.final_output,
            "official_result": self.official_result.to_dict(),
            "oracle_result": self.oracle_result.to_dict(),
            "exploit_labels": serialize_value(self.exploit_labels),
            "runtime": self.runtime.to_dict(),
            "steps": serialize_value([step.to_dict() for step in self.steps]),
            "annotations": serialize_value(self.annotations),
            "verifier_gap": self.verifier_gap,
            "is_false_pass": self.is_false_pass,
        }
