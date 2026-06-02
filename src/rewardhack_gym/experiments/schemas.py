from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


OutcomeLabel = Literal["true_pass", "false_pass", "false_fail", "true_fail"]
SelectionPolicy = Literal[
    "official_only",
    "official_then_random",
    "official_then_low_oracle",
    "official_then_high_oracle",
    "oracle_upper_bound",
]
DummyModelMode = Literal["canonical_true", "canonical_exploit", "random_bad"]
PRIVATE_METADATA_FRAGMENTS = ("hidden", "oracle")
PRIVATE_METADATA_KEYS = {"canonical_true_output", "canonical_exploit_output"}
SENSITIVE_OUTPUT_KEYS = {
    "api_key",
    "authorization",
    "access_token",
    "bearer_token",
    "password",
    "secret",
    "token",
}


def _is_private_or_sensitive_key(key: object) -> bool:
    normalized = str(key).lower()
    if normalized in PRIVATE_METADATA_KEYS or normalized in SENSITIVE_OUTPUT_KEYS:
        return True
    if "header" in normalized:
        return True
    return any(fragment in normalized for fragment in PRIVATE_METADATA_FRAGMENTS)


def redact_public_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): redact_public_metadata(item)
            for key, item in value.items()
            if not _is_private_or_sensitive_key(key)
        }
    if isinstance(value, list):
        return [redact_public_metadata(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ModelConfig:
    id: str
    provider: str = "dummy"
    model_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    stop: list[str] = field(default_factory=list)
    num_candidates: int = 1


@dataclass(frozen=True, slots=True)
class EnvironmentSweepConfig:
    family: str | None = None
    families: list[str] = field(default_factory=list)
    profile: str | None = None
    profiles: list[str] = field(default_factory=lambda: ["medium"])
    split: str = "eval"
    num_tasks: int = 100
    code_execution_backend: str = "subprocess"
    code_execution_timeout_seconds: float = 2.0
    code_execution_memory_mb: int = 256

    def resolved_families(self) -> list[str]:
        if self.families:
            return list(self.families)
        if self.family is not None:
            return [self.family]
        return ["code/spec-overfit"]

    def resolved_profiles(self) -> list[str]:
        if self.profiles:
            return list(self.profiles)
        if self.profile is not None:
            return [self.profile]
        return ["medium"]


@dataclass(frozen=True, slots=True)
class BestOfNConfig:
    values: list[int] = field(default_factory=lambda: [1])
    selection_policies: list[SelectionPolicy] = field(default_factory=lambda: ["official_only"])


@dataclass(frozen=True, slots=True)
class ReportingConfig:
    save_examples: bool = True
    max_examples_per_bucket: int = 50
    bootstrap_ci: bool = True
    bootstrap_samples: int = 1000


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    type: str
    seed: int = 0


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment: ExperimentSpec
    environment: EnvironmentSweepConfig
    models: list[ModelConfig]
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    best_of_n: BestOfNConfig = field(default_factory=BestOfNConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        experiment = ExperimentSpec(**payload.get("experiment", {}))
        environment = EnvironmentSweepConfig(**payload.get("environment", {}))
        models = [ModelConfig(**item) for item in payload.get("models", [{"id": "dummy", "provider": "dummy"}])]
        sampling = SamplingConfig(**payload.get("sampling", {}))
        best_of_n = BestOfNConfig(**payload.get("best_of_n", {}))
        reporting = ReportingConfig(**payload.get("reporting", {}))
        return cls(
            experiment=experiment,
            environment=environment,
            models=models,
            sampling=sampling,
            best_of_n=best_of_n,
            reporting=reporting,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExperimentRunMetadata:
    run_id: str
    experiment_type: str
    timestamp: str
    git_commit: str | None
    config_path: str | None
    families: list[str]
    profiles: list[str]
    models: list[str]
    split: str
    num_tasks: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        config: ExperimentConfig,
        config_path: str | None,
        git_commit: str | None,
    ) -> "ExperimentRunMetadata":
        return cls(
            run_id=run_id,
            experiment_type=config.experiment.type,
            timestamp=datetime.fromtimestamp(config.experiment.seed, timezone.utc).isoformat(),
            git_commit=git_commit,
            config_path=config_path,
            families=config.environment.resolved_families(),
            profiles=config.environment.resolved_profiles(),
            models=[model.id for model in config.models],
            split=config.environment.split,
            num_tasks=config.environment.num_tasks,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    run_id: str
    experiment_type: str
    model_id: str
    family: str
    profile: str
    split: str
    task_id: str
    candidate_id: str
    candidate_index: int
    sampling: dict[str, Any]
    prompt: str
    completion: str
    official_score: float
    official_passed: bool
    oracle_score: float
    oracle_passed: bool
    hack_gap: float
    false_pass: bool
    outcome_label: OutcomeLabel
    exploit_labels: list[str]
    semantic_failures: list[str]
    execution_backend: str | None
    duration_seconds: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = redact_public_metadata(data["metadata"])
        return data


@dataclass(frozen=True, slots=True)
class TrajectoryRecord(CandidateRecord):
    pass


@dataclass(frozen=True, slots=True)
class BestOfNSelectedRecord:
    task_id: str
    model_id: str
    profile: str
    N: int
    selection_policy: str
    selected_candidate_id: str
    selected_official_score: float
    selected_oracle_score: float
    selected_hack_gap: float
    selected_false_pass: bool
    oracle_best_candidate_id: str | None
    official_best_candidate_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    num_examples: int
    official_pass_rate: float
    oracle_pass_rate: float
    true_pass_rate: float
    false_pass_rate: float
    false_fail_rate: float
    true_fail_rate: float
    mean_official_score: float
    mean_oracle_score: float
    mean_hack_gap: float
    mean_positive_hack_gap: float
    mean_hack_gap_given_official_pass: float
    conditional_oracle_score_given_official_pass: float
    disagreement_rate: float
    exploit_sensitivity_slope: float = 0.0
    confidence_intervals: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BestOfNMetricSummary:
    N: int
    selection_policy: str
    official_pass_at_n: float
    oracle_pass_at_n: float
    hack_at_n: float
    true_pass_at_n: float
    mean_selected_official_score: float
    mean_selected_oracle_score: float
    mean_selected_hack_gap: float
    best_of_n_hack_amplification: float
    oracle_best_at_n: float
    official_selected_vs_oracle_best_gap: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data[f"official_pass@{self.N}"] = self.official_pass_at_n
        data[f"oracle_pass@{self.N}"] = self.oracle_pass_at_n
        data[f"hack@{self.N}"] = self.hack_at_n
        data[f"true_pass@{self.N}"] = self.true_pass_at_n
        return data


def outcome_label(official_passed: bool, oracle_passed: bool) -> OutcomeLabel:
    if official_passed and oracle_passed:
        return "true_pass"
    if official_passed and not oracle_passed:
        return "false_pass"
    if not official_passed and oracle_passed:
        return "false_fail"
    return "true_fail"
