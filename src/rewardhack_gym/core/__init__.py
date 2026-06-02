"""Core abstractions for RewardHack-Gym."""

from rewardhack_gym.core.base import Checker, CompositeEvaluator, FunctionalChecker, ResearchEnvironment
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
    "Checker",
    "CheckerResult",
    "CompositeEvaluator",
    "EnvironmentConfig",
    "EvaluatorResult",
    "ExploitSurface",
    "ExploitabilityProfile",
    "FunctionalChecker",
    "ResearchEnvironment",
    "RuntimeMetadata",
    "ENVIRONMENT_VERSION",
    "GENERATOR_VERSION",
    "OFFICIAL_VERIFIER_VERSION",
    "ORACLE_VERIFIER_VERSION",
    "SUPPORTED_CODE_EXECUTION_BACKENDS",
    "SUPPORTED_DATASET_SPLITS",
    "Task",
    "TASK_SCHEMA_VERSION",
    "Trajectory",
    "TrajectoryStep",
    "create_environment",
    "list_environments",
    "register_environment",
]
