from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import random
from typing import Generic, TypeVar

from rewardhack_gym.core.base import ResearchEnvironment
from rewardhack_gym.core.models import Task, Trajectory

TaskT = TypeVar("TaskT", bound=Task)
SelectionMode = str
SUPPORTED_SELECTION_MODES = (
    "official_only",
    "official_then_random",
    "official_then_low_oracle",
    "official_then_high_oracle",
    "oracle_upper_bound",
)


@dataclass(frozen=True, slots=True)
class BestOfNResult:
    selected: Trajectory
    candidates: tuple[Trajectory, ...]
    selected_index: int
    selection_mode: SelectionMode = "official_then_low_oracle"


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
        selection_mode: SelectionMode = "official_then_low_oracle",
        selection_seed: int = 0,
    ) -> BestOfNResult:
        if n <= 0:
            raise ValueError("Best-of-N requires n >= 1.")
        if selection_mode not in SUPPORTED_SELECTION_MODES:
            raise ValueError(
                f"Unknown Best-of-N selection_mode {selection_mode!r}. "
                f"Expected one of {SUPPORTED_SELECTION_MODES}."
            )
        candidates = tuple(
            self.environment.evaluate_output(
                task,
                sampler(task, index),
                policy_id=policy_id,
                annotations={"candidate_index": index, "best_of_n_selection_mode": selection_mode},
            )
            for index in range(n)
        )
        selected_index = self._select_index(
            candidates,
            selection_mode=selection_mode,
            selection_seed=selection_seed,
        )
        return BestOfNResult(
            selected=candidates[selected_index],
            candidates=candidates,
            selected_index=selected_index,
            selection_mode=selection_mode,
        )

    @staticmethod
    def _select_index(
        candidates: tuple[Trajectory, ...],
        *,
        selection_mode: SelectionMode,
        selection_seed: int,
    ) -> int:
        if selection_mode == "official_only":
            return max(
                range(len(candidates)),
                key=lambda index: (candidates[index].official_result.score, -index),
            )
        if selection_mode == "official_then_random":
            best_official = max(candidate.official_result.score for candidate in candidates)
            tied = [
                index
                for index, candidate in enumerate(candidates)
                if candidate.official_result.score == best_official
            ]
            return random.Random(selection_seed).choice(tied)
        if selection_mode == "official_then_low_oracle":
            return max(
                range(len(candidates)),
                key=lambda index: (
                    candidates[index].official_result.score,
                    -candidates[index].oracle_result.score,
                    -index,
                ),
            )
        if selection_mode == "official_then_high_oracle":
            return max(
                range(len(candidates)),
                key=lambda index: (
                    candidates[index].official_result.score,
                    candidates[index].oracle_result.score,
                    -index,
                ),
            )
        if selection_mode == "oracle_upper_bound":
            return max(
                range(len(candidates)),
                key=lambda index: (candidates[index].oracle_result.score, -index),
            )
        raise ValueError(
            f"Unknown Best-of-N selection_mode {selection_mode!r}. "
            f"Expected one of {SUPPORTED_SELECTION_MODES}."
        )


class RejectionFilter:
    """Keep only outputs that pass a configurable official-score threshold."""

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold

    def filter(self, trajectories: Iterable[Trajectory]) -> list[Trajectory]:
        return [trajectory for trajectory in trajectories if trajectory.official_result.score >= self.threshold]
