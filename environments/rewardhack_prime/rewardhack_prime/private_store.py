from __future__ import annotations

from rewardhack_gym import Task


class PrivateTaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add(self, task: Task) -> None:
        self._tasks[task.task_id] = task

    def get(self, task_id: str) -> Task:
        return self._tasks[task_id]

    def __contains__(self, task_id: object) -> bool:
        return isinstance(task_id, str) and task_id in self._tasks
