from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Generic, TypeVar

from rewardhack_gym.core.base import ResearchEnvironment
from rewardhack_gym.core.models import Task, Trajectory

TaskT = TypeVar("TaskT", bound=Task)


class PromptRunner(Generic[TaskT]):
    """Runs a callable model/agent against tasks and returns trajectories."""

    def __init__(self, environment: ResearchEnvironment[TaskT]) -> None:
        self.environment = environment

    def run_task(
        self,
        task: TaskT,
        agent: Callable[[TaskT], str],
        *,
        policy_id: str | None = None,
        annotations: dict[str, object] | None = None,
    ) -> Trajectory:
        output = agent(task)
        return self.environment.evaluate_output(
            task,
            output,
            policy_id=policy_id,
            annotations=annotations,
        )

    def run_many(
        self,
        tasks: Iterable[TaskT],
        agent: Callable[[TaskT], str],
        *,
        policy_id: str | None = None,
    ) -> list[Trajectory]:
        return [self.run_task(task, agent, policy_id=policy_id) for task in tasks]

    async def run_many_async(
        self,
        tasks: Sequence[TaskT],
        agent: Callable[[TaskT], Awaitable[str]],
        *,
        policy_id: str | None = None,
    ) -> list[Trajectory]:
        outputs = await asyncio.gather(*(agent(task) for task in tasks))
        return [
            self.environment.evaluate_output(task, output, policy_id=policy_id)
            for task, output in zip(tasks, outputs, strict=True)
        ]

