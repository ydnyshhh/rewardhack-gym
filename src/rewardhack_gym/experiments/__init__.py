"""Experiment-level runners for RewardHack-Gym research sweeps."""

from rewardhack_gym.experiments.metrics import (
    compute_best_of_n_metrics,
    compute_candidate_metrics,
)
from rewardhack_gym.experiments.model_clients import OpenAICompatibleClient, build_model_client
from rewardhack_gym.experiments.reporting import generate_report
from rewardhack_gym.experiments.runners import (
    run_best_of_n_experiment,
    run_model_sweep_experiment,
    run_profile_sweep_experiment,
)
from rewardhack_gym.experiments.schemas import (
    BestOfNMetricSummary,
    BestOfNSelectedRecord,
    CandidateRecord,
    EnvironmentSweepConfig,
    ExperimentConfig,
    ExperimentRunMetadata,
    MetricSummary,
    ModelConfig,
    SamplingConfig,
    TrajectoryRecord,
)

__all__ = [
    "BestOfNMetricSummary",
    "BestOfNSelectedRecord",
    "CandidateRecord",
    "EnvironmentSweepConfig",
    "ExperimentConfig",
    "ExperimentRunMetadata",
    "MetricSummary",
    "ModelConfig",
    "SamplingConfig",
    "TrajectoryRecord",
    "OpenAICompatibleClient",
    "build_model_client",
    "compute_best_of_n_metrics",
    "compute_candidate_metrics",
    "generate_report",
    "run_best_of_n_experiment",
    "run_model_sweep_experiment",
    "run_profile_sweep_experiment",
]
