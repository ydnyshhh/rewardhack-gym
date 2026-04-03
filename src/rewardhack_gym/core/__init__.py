"""Core abstractions for RewardHack-Gym."""

from rewardhack_gym.core.base import Checker, CompositeEvaluator, FunctionalChecker, ResearchEnvironment
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
    "Task",
    "Trajectory",
    "TrajectoryStep",
    "create_environment",
    "list_environments",
    "register_environment",
]
