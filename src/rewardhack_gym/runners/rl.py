from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from rewardhack_gym.core.base import ResearchEnvironment
from rewardhack_gym.core.models import Task, Trajectory

TaskT = TypeVar("TaskT", bound=Task)


@dataclass(frozen=True, slots=True)
class RewardRecord:
    official_reward: float
    oracle_reward: float
    verifier_gap: float
    false_pass: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class RewardAdapter(Generic[TaskT]):
    """Expose environment scores as reward-like records for external RL systems."""

    def __init__(self, environment: ResearchEnvironment[TaskT]) -> None:
        self.environment = environment

    def score_output(
        self,
        task: TaskT,
        output: str,
        *,
        policy_id: str | None = None,
    ) -> tuple[Trajectory, RewardRecord]:
        trajectory = self.environment.evaluate_output(task, output, policy_id=policy_id)
        return trajectory, self.from_trajectory(trajectory)

    @staticmethod
    def from_trajectory(trajectory: Trajectory) -> RewardRecord:
        return RewardRecord(
            official_reward=trajectory.official_result.score,
            oracle_reward=trajectory.oracle_result.score,
            verifier_gap=trajectory.verifier_gap,
            false_pass=trajectory.is_false_pass,
            metadata={
                "exploit_labels": list(trajectory.exploit_labels),
                "task_id": trajectory.task.task_id,
                "family": trajectory.task.family,
            },
        )


class AsyncRewardAdapter(RewardAdapter[TaskT]):
    """Reward adapter that records extra metadata commonly needed in async RL pipelines."""

    def score_output(
        self,
        task: TaskT,
        output: str,
        *,
        policy_id: str | None = None,
        rollout_version_id: str | None = None,
        evaluator_version_id: str | None = None,
    ) -> tuple[Trajectory, RewardRecord]:
        trajectory = self.environment.evaluate_output(
            task,
            output,
            policy_id=policy_id,
            rollout_version_id=rollout_version_id,
            evaluator_version_id=evaluator_version_id,
            annotations={"async_rl": True},
        )
        record = self.from_trajectory(trajectory)
        record.metadata.update(
            {
                "policy_id": policy_id,
                "rollout_version_id": rollout_version_id,
                "evaluator_version_id": evaluator_version_id,
                "environment_profile": self.environment.config.exploitability.level,
            }
        )
        return trajectory, record
