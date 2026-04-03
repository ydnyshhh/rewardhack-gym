"""RewardHack-Gym public API."""

from rewardhack_gym.analysis.metrics import MetricSummary, summarize_trajectories
from rewardhack_gym.bootstrap import bootstrap_builtin_environments
from rewardhack_gym.core.base import ResearchEnvironment
from rewardhack_gym.core.config import EnvironmentConfig, ExploitabilityProfile
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

__all__ = [
    "CheckerResult",
    "EnvironmentConfig",
    "EvaluatorResult",
    "ExploitSurface",
    "ExploitabilityProfile",
    "MetricSummary",
    "ResearchEnvironment",
    "RuntimeMetadata",
    "Task",
    "Trajectory",
    "TrajectoryStep",
    "bootstrap_builtin_environments",
    "create_environment",
    "list_environments",
    "register_environment",
    "summarize_trajectories",
]
