from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from rewardhack_gym.core.base import ResearchEnvironment
from rewardhack_gym.core.models import Task, Trajectory

TaskT = TypeVar("TaskT", bound=Task)


@dataclass(frozen=True, slots=True)
class BestOfNResult:
    selected: Trajectory
    candidates: tuple[Trajectory, ...]
    selected_index: int


class BestOfNRunner(Generic[TaskT]):
    """Sample multiple outputs, select under the official verifier, then inspect the oracle gap."""

    def __init__(self, environment: ResearchEnvironment[TaskT]) -> None:
        self.environment = environment

    def run(
        self,
        task: TaskT,
        sampler: Callable[[TaskT, int], str],
        *,
        n: int,
        policy_id: str | None = None,
    ) -> BestOfNResult:
        if n <= 0:
            raise ValueError("Best-of-N requires n >= 1.")
        candidates = tuple(
            self.environment.evaluate_output(
                task,
                sampler(task, index),
                policy_id=policy_id,
                annotations={"candidate_index": index},
            )
            for index in range(n)
        )
        scored = [
            (candidate.official_result.score, -candidate.oracle_result.score, index, candidate)
            for index, candidate in enumerate(candidates)
        ]
        _, _, selected_index, selected = max(scored, key=lambda item: item[:3])
        return BestOfNResult(selected=selected, candidates=candidates, selected_index=selected_index)


class RejectionFilter:
    """Keep only outputs that pass a configurable official-score threshold."""

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold

    def filter(self, trajectories: Iterable[Trajectory]) -> list[Trajectory]:
        return [trajectory for trajectory in trajectories if trajectory.official_result.score >= self.threshold]

