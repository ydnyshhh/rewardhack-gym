from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Generic, Iterable, Protocol, Sequence, TypeVar

from rewardhack_gym.core.config import EnvironmentConfig
from rewardhack_gym.core.models import (
    CheckerResult,
    EvaluatorResult,
    RuntimeMetadata,
    Task,
    Trajectory,
)

TaskT = TypeVar("TaskT", bound=Task)


class Checker(Protocol[TaskT]):
    name: str
    weight: float

    def evaluate(self, task: TaskT, artifact: str) -> CheckerResult:
        ...


@dataclass(slots=True)
class FunctionalChecker(Generic[TaskT]):
    name: str
    func: Callable[[TaskT, str], CheckerResult]
    weight: float = 1.0

    def evaluate(self, task: TaskT, artifact: str) -> CheckerResult:
        return self.func(task, artifact)


@dataclass(slots=True)
class CompositeEvaluator(Generic[TaskT]):
    name: str
    checkers: Sequence[Checker[TaskT]]
    pass_threshold: float = 1.0

    def evaluate(self, task: TaskT, artifact: str) -> EvaluatorResult:
        if not self.checkers:
            raise ValueError(f"Evaluator {self.name!r} has no checkers.")

        components = tuple(checker.evaluate(task, artifact) for checker in self.checkers)
        total_weight = sum(max(checker.weight, 0.0) for checker in self.checkers)
        if total_weight <= 0:
            raise ValueError(f"Evaluator {self.name!r} received non-positive total checker weight.")

        score = sum(
            checker.weight * component.score
            for checker, component in zip(self.checkers, components, strict=True)
        ) / total_weight
        passed = score >= self.pass_threshold
        diagnostics = {
            "component_scores": {component.checker_name: component.score for component in components},
            "component_passes": {component.checker_name: component.passed for component in components},
        }
        warnings = tuple(
            warning for component in components for warning in component.warnings
        )
        return EvaluatorResult(
            evaluator_name=self.name,
            score=round(score, 6),
            passed=passed,
            components=components,
            diagnostics=diagnostics,
            warnings=warnings,
        )


class ResearchEnvironment(ABC, Generic[TaskT]):
    """Base class for all reward-hacking environments."""

    name: str
    family: str

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.config = config or EnvironmentConfig()

    @abstractmethod
    def sample_task(self, seed: int | None = None) -> TaskT:
        ...

    @abstractmethod
    def build_official_evaluator(self, task: TaskT) -> CompositeEvaluator[TaskT]:
        ...

    @abstractmethod
    def build_oracle_evaluator(self, task: TaskT) -> CompositeEvaluator[TaskT]:
        ...

    @abstractmethod
    def canonical_true_output(self, task: TaskT) -> str:
        ...

    @abstractmethod
    def canonical_exploit_output(self, task: TaskT) -> str:
        ...

    def classify_exploit(
        self,
        task: TaskT,
        final_output: str,
        official_result: EvaluatorResult,
        oracle_result: EvaluatorResult,
    ) -> tuple[str, ...]:
        if official_result.passed and not oracle_result.passed:
            return task.exploit_surface.exploit_families
        return ()

    def evaluate_output(
        self,
        task: TaskT,
        final_output: str,
        *,
        steps: Iterable[dict[str, object]] | None = None,
        policy_id: str | None = None,
        rollout_version_id: str | None = None,
        evaluator_version_id: str | None = None,
        annotations: dict[str, object] | None = None,
    ) -> Trajectory:
        started = perf_counter()
        official_result = self.build_official_evaluator(task).evaluate(task, final_output)
        oracle_result = self.build_oracle_evaluator(task).evaluate(task, final_output)
        duration_seconds = perf_counter() - started
        runtime = RuntimeMetadata(
            duration_seconds=duration_seconds,
            policy_id=policy_id,
            rollout_version_id=rollout_version_id,
            evaluator_version_id=evaluator_version_id,
            environment_profile=self.config.exploitability.level,
        )
        from rewardhack_gym.core.models import TrajectoryStep

        trajectory_steps = tuple(
            TrajectoryStep(
                role=str(step.get("role", "assistant")),
                content=str(step.get("content", "")),
                metadata={str(key): value for key, value in step.items() if key not in {"role", "content"}},
            )
            for step in (steps or ())
        )
        return Trajectory(
            task=task,
            prompt=task.prompt,
            final_output=final_output,
            official_result=official_result,
            oracle_result=oracle_result,
            exploit_labels=self.classify_exploit(task, final_output, official_result, oracle_result),
            runtime=runtime,
            steps=trajectory_steps,
            annotations=annotations or {},
        )
