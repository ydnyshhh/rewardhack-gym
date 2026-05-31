from __future__ import annotations

from collections.abc import Mapping, MutableMapping

import verifiers as vf

from rewardhack_gym import EnvironmentConfig, ResearchEnvironment, Task, Trajectory, create_environment
from rewardhack_prime.config import RewardHackTasksetConfig, SUPPORTED_CODE_EXECUTION_BACKENDS, SUPPORTED_PROFILES
from rewardhack_prime.conversion import rewardhack_task_to_vf_task
from rewardhack_prime.private_store import PrivateTaskStore
from rewardhack_prime.scoring import (
    SUPPORTED_REWARD_MODES,
    completion_text_from_state,
    reward_from_trajectory,
    task_id_from_vf_task,
    trajectory_steps_from_state,
)


class RewardHackTaskset(vf.Taskset):
    config_type = RewardHackTasksetConfig

    def __init__(self, config: RewardHackTasksetConfig) -> None:
        super().__init__(config=config)
        self._env: ResearchEnvironment[Task] | None = None
        self.private_store = PrivateTaskStore()
        self._public_rows_cache: list[dict[str, object]] | None = None
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
        if self.config.reward_penalty_lambda < 0:
            raise ValueError("RewardHackTasksetConfig.reward_penalty_lambda must be non-negative.")
        if self.config.code_execution_backend not in SUPPORTED_CODE_EXECUTION_BACKENDS:
            raise ValueError(
                f"Unknown RewardHack code_execution_backend {self.config.code_execution_backend!r}. "
                f"Expected one of {SUPPORTED_CODE_EXECUTION_BACKENDS}."
            )
        if self.config.code_execution_timeout_seconds <= 0:
            raise ValueError("RewardHackTasksetConfig.code_execution_timeout_seconds must be positive.")
        if self.config.code_execution_memory_mb <= 0:
            raise ValueError("RewardHackTasksetConfig.code_execution_memory_mb must be positive.")
        if self.config.code_execution_stdout_limit_chars <= 0:
            raise ValueError("RewardHackTasksetConfig.code_execution_stdout_limit_chars must be positive.")
        if self.config.code_execution_stderr_limit_chars <= 0:
            raise ValueError("RewardHackTasksetConfig.code_execution_stderr_limit_chars must be positive.")
        if self.config.code_execution_max_output_object_size <= 0:
            raise ValueError("RewardHackTasksetConfig.code_execution_max_output_object_size must be positive.")
        if self.config.prime_sandbox_timeout_minutes <= 0:
            raise ValueError("RewardHackTasksetConfig.prime_sandbox_timeout_minutes must be positive.")
        if self.config.prime_sandbox_cpu_cores <= 0:
            raise ValueError("RewardHackTasksetConfig.prime_sandbox_cpu_cores must be positive.")

    @property
    def rewardhack_env(self) -> ResearchEnvironment[Task]:
        if self._env is None:
            env_config = EnvironmentConfig.from_profile(
                seed=self.config.seed,
                profile=self.config.profile,
                max_runtime_seconds=self.config.code_execution_timeout_seconds,
                code_execution_backend=self.config.code_execution_backend,
                code_execution_timeout_seconds=self.config.code_execution_timeout_seconds,
                code_execution_memory_mb=self.config.code_execution_memory_mb,
                code_execution_stdout_limit_chars=self.config.code_execution_stdout_limit_chars,
                code_execution_stderr_limit_chars=self.config.code_execution_stderr_limit_chars,
                code_execution_max_output_object_size=self.config.code_execution_max_output_object_size,
                prime_sandbox_image=self.config.prime_sandbox_image,
                prime_sandbox_timeout_minutes=self.config.prime_sandbox_timeout_minutes,
                prime_sandbox_cpu_cores=self.config.prime_sandbox_cpu_cores,
                metadata={
                    "prime_split": self.config.split,
                    "prime_reward_mode": self.config.reward_mode,
                    "prime_reward_penalty_lambda": self.config.reward_penalty_lambda,
                    "code_execution_backend": self.config.code_execution_backend,
                },
            )
            self._env = create_environment(self.config.family, env_config)
        return self._env

    def _sample_unique_task(self, index: int) -> Task:
        base_seed = self.config.seed + index
        for attempt in range(100):
            task = self.rewardhack_env.sample_task(seed=base_seed + attempt * 1_000_003)
            if task.task_id not in self.private_store:
                return task
        raise RuntimeError(f"Could not sample a unique RewardHack task for index {index}.")

    def _rewardhack_rows(self) -> list[dict[str, object]]:
        if self._public_rows_cache is not None:
            return self._public_rows_cache

        rows: list[dict[str, object]] = []
        for index in range(self.config.num_tasks):
            task = self._sample_unique_task(index)
            self.private_store.add(task)
            row = rewardhack_task_to_vf_task(
                task,
                include_canonical_outputs=self.config.expose_canonical_outputs,
            )
            row["split"] = self.config.split
            row["max_turns"] = 1
            rows.append(row)
        self._public_rows_cache = rows
        return rows

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if str(split) != self.config.split:
            return []
        return self._rewardhack_rows()

    def load_eval_tasks(self) -> vf.Tasks:
        if self.config.split != "eval":
            return []
        return self._rewardhack_rows()

    def rows(self) -> vf.Tasks:
        if self.config.split == "eval":
            return []
        return self._rewardhack_rows()

    def eval_rows(self) -> vf.Tasks:
        if self.config.split != "eval":
            return []
        return self._rewardhack_rows()

    async def run_rewardhack_eval(
        self,
        task: Mapping[str, object],
        completion: object,
        state: Mapping[str, object],
    ) -> Trajectory:
        task_id = task_id_from_vf_task(task)
        if task_id not in self.private_store:
            self._rewardhack_rows()
        rewardhack_task = self.private_store.get(task_id)
        return self.rewardhack_env.evaluate_output(
            rewardhack_task,
            completion_text_from_state({"completion": completion}),
            steps=trajectory_steps_from_state(state),
        )

    async def score_once(
        self,
        task: Mapping[str, object],
        completion: object,
        state: Mapping[str, object],
    ) -> Trajectory:
        cached = state.get("rewardhack_trajectory")
        if isinstance(cached, Trajectory):
            return cached

        trajectory = await self.run_rewardhack_eval(task, completion, state)
        if isinstance(state, MutableMapping):
            state["rewardhack_trajectory"] = trajectory
        return trajectory

    @vf.reward(weight=1.0)
    async def official_reward(self, task: vf.Task, state: vf.State) -> float:
        trajectory = await self.score_once(task, state.get("completion", []), state)
        return reward_from_trajectory(
            trajectory,
            self.config.reward_mode,
            penalty_lambda=self.config.reward_penalty_lambda,
        )

    @vf.reward(weight=0.0)
    async def oracle_score(self, task: vf.Task, state: vf.State) -> float:
        if not self.config.include_oracle_metrics:
            return 0.0
        return float((await self.score_once(task, state.get("completion", []), state)).oracle_result.score)

    @vf.reward(weight=0.0)
    async def hack_gap(self, task: vf.Task, state: vf.State) -> float:
        if not self.config.include_oracle_metrics:
            return 0.0
        trajectory = await self.score_once(task, state.get("completion", []), state)
        return float(trajectory.official_result.score - trajectory.oracle_result.score)

    @vf.reward(weight=0.0)
    async def false_pass(self, task: vf.Task, state: vf.State) -> float:
        if not self.config.include_oracle_metrics:
            return 0.0
        return float((await self.score_once(task, state.get("completion", []), state)).is_false_pass)


def load_taskset(config: RewardHackTasksetConfig) -> RewardHackTaskset:
    return RewardHackTaskset(config=config)
