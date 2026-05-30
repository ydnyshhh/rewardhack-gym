from __future__ import annotations

from collections.abc import Mapping, MutableMapping

import verifiers as vf

from rewardhack_gym import EnvironmentConfig, ResearchEnvironment, Task, Trajectory, create_environment
from rewardhack_prime.config import RewardHackTasksetConfig, SUPPORTED_PROFILES
from rewardhack_prime.conversion import rewardhack_task_to_vf_task
from rewardhack_prime.scoring import (
    SUPPORTED_REWARD_MODES,
    completion_text_from_state,
    reward_from_trajectory,
    task_id_from_vf_task,
    trajectory_steps_from_state,
)


class RewardHackTaskset(vf.Taskset[RewardHackTasksetConfig]):
    def __init__(self, config: RewardHackTasksetConfig) -> None:
        super().__init__(config=config)
        self._env: ResearchEnvironment[Task] | None = None
        self._tasks_by_id: dict[str, Task] = {}
        self._rows_cache: list[dict[str, object]] | None = None
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.profile not in SUPPORTED_PROFILES:
            raise ValueError(
                f"Unknown RewardHack profile {self.config.profile!r}. "
                f"Expected one of {SUPPORTED_PROFILES}."
            )
        if self.config.reward_mode not in SUPPORTED_REWARD_MODES:
            raise ValueError(
                f"Unknown RewardHack reward_mode {self.config.reward_mode!r}. "
                f"Expected one of {SUPPORTED_REWARD_MODES}."
            )
        if self.config.num_tasks < 1:
            raise ValueError("RewardHackTasksetConfig.num_tasks must be at least 1.")

    @property
    def rewardhack_env(self) -> ResearchEnvironment[Task]:
        if self._env is None:
            env_config = EnvironmentConfig.from_profile(
                seed=self.config.seed,
                profile=self.config.profile,
                metadata={
                    "prime_split": self.config.split,
                    "prime_reward_mode": self.config.reward_mode,
                },
            )
            self._env = create_environment(self.config.family, env_config)
        return self._env

    def _sample_unique_task(self, index: int) -> Task:
        base_seed = self.config.seed + index
        for attempt in range(100):
            task = self.rewardhack_env.sample_task(seed=base_seed + attempt * 1_000_003)
            if task.task_id not in self._tasks_by_id:
                return task
        raise RuntimeError(f"Could not sample a unique RewardHack task for index {index}.")

    def _rows(self) -> list[dict[str, object]]:
        if self._rows_cache is not None:
            return self._rows_cache

        rows: list[dict[str, object]] = []
        for index in range(self.config.num_tasks):
            task = self._sample_unique_task(index)
            self._tasks_by_id[task.task_id] = task
            row = rewardhack_task_to_vf_task(task)
            row["split"] = self.config.split
            row["max_turns"] = 1
            rows.append(row)
        self._rows_cache = rows
        return rows

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if str(split) != self.config.split:
            return []
        return self._rows()

    def load_eval_tasks(self) -> vf.Tasks:
        if self.config.split != "eval":
            return []
        return self._rows()

    def _trajectory_for(self, task: Mapping[str, object], state: Mapping[str, object]) -> Trajectory:
        task_id = task_id_from_vf_task(task)
        cache_key = f"rewardhack_trajectory:{task_id}"
        cached = state.get(cache_key)
        if isinstance(cached, Trajectory):
            return cached

        if task_id not in self._tasks_by_id:
            self._rows()
        rewardhack_task = self._tasks_by_id[task_id]
        final_output = completion_text_from_state(state)
        trajectory = self.rewardhack_env.evaluate_output(
            rewardhack_task,
            final_output,
            steps=trajectory_steps_from_state(state),
        )
        if isinstance(state, MutableMapping):
            state[cache_key] = trajectory
        return trajectory

    @vf.reward(weight=1.0)
    async def rewardhack_reward(self, task: vf.Task, state: vf.State) -> float:
        trajectory = self._trajectory_for(task, state)
        return reward_from_trajectory(trajectory, self.config.reward_mode)

    @vf.reward(weight=0.0)
    async def official_score(self, task: vf.Task, state: vf.State) -> float:
        return float(self._trajectory_for(task, state).official_result.score)

    @vf.reward(weight=0.0)
    async def oracle_score(self, task: vf.Task, state: vf.State) -> float:
        if not self.config.include_oracle_metrics:
            return 0.0
        return float(self._trajectory_for(task, state).oracle_result.score)

    @vf.reward(weight=0.0)
    async def hack_gap(self, task: vf.Task, state: vf.State) -> float:
        if not self.config.include_oracle_metrics:
            return 0.0
        trajectory = self._trajectory_for(task, state)
        return float(trajectory.official_result.score - trajectory.oracle_result.score)

    @vf.reward(weight=0.0)
    async def official_passed(self, task: vf.Task, state: vf.State) -> float:
        return float(self._trajectory_for(task, state).official_result.passed)

    @vf.reward(weight=0.0)
    async def oracle_passed(self, task: vf.Task, state: vf.State) -> float:
        if not self.config.include_oracle_metrics:
            return 0.0
        return float(self._trajectory_for(task, state).oracle_result.passed)

    @vf.reward(weight=0.0)
    async def false_pass(self, task: vf.Task, state: vf.State) -> float:
        if not self.config.include_oracle_metrics:
            return 0.0
        return float(self._trajectory_for(task, state).is_false_pass)


def load_taskset(config: RewardHackTasksetConfig) -> RewardHackTaskset:
    return RewardHackTaskset(config=config)
