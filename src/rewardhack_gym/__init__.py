"""RewardHack-Gym public API."""

from rewardhack_gym.analysis.mech_interp import (
    MechInterpRecord,
    build_matched_pairs,
    build_mech_interp_record,
    build_mech_interp_records,
)
from rewardhack_gym.analysis.metrics import BestOfNMetricSummary, MetricSummary, summarize_best_of_n, summarize_trajectories
from rewardhack_gym.bootstrap import bootstrap_builtin_environments
from rewardhack_gym.core.base import ResearchEnvironment
from rewardhack_gym.core.config import EnvironmentConfig, ExploitabilityProfile, SUPPORTED_CODE_EXECUTION_BACKENDS
from rewardhack_gym.core.models import (
    CheckerResult,
    EvaluatorResult,
    ExploitSurface,
    RuntimeMetadata,
    Task,
    Trajectory,
    TrajectoryStep,
)
from rewardhack_gym.core.registry import create_environment, list_environments, register_environment
from rewardhack_gym.core.splits import SUPPORTED_DATASET_SPLITS
from rewardhack_gym.core.versions import (
    ENVIRONMENT_VERSION,
    GENERATOR_VERSION,
    OFFICIAL_VERIFIER_VERSION,
    ORACLE_VERIFIER_VERSION,
    TASK_SCHEMA_VERSION,
)

__all__ = [
    "CheckerResult",
    "EnvironmentConfig",
    "EvaluatorResult",
    "ExploitSurface",
    "ExploitabilityProfile",
    "MetricSummary",
    "BestOfNMetricSummary",
    "MechInterpRecord",
    "ENVIRONMENT_VERSION",
    "GENERATOR_VERSION",
    "OFFICIAL_VERIFIER_VERSION",
    "ORACLE_VERIFIER_VERSION",
    "ResearchEnvironment",
    "RuntimeMetadata",
    "SUPPORTED_CODE_EXECUTION_BACKENDS",
    "SUPPORTED_DATASET_SPLITS",
    "Task",
    "TASK_SCHEMA_VERSION",
    "Trajectory",
    "TrajectoryStep",
    "build_matched_pairs",
    "build_mech_interp_record",
    "build_mech_interp_records",
    "bootstrap_builtin_environments",
    "create_environment",
    "list_environments",
    "register_environment",
    "summarize_best_of_n",
    "summarize_trajectories",
]
