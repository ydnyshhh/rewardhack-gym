from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from rewardhack_gym import create_environment


def contains_key(value: object, key: str) -> bool:
    if isinstance(value, Mapping):
        return any(
            item_key == key or contains_key(item_value, key)
            for item_key, item_value in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return any(contains_key(item, key) for item in value)
    return False


class FakeTasksetConfig:
    def __init__(self, **kwargs: Any) -> None:
        for cls in reversed(type(self).mro()):
            for name in getattr(cls, "__annotations__", {}):
                if hasattr(cls, name):
                    setattr(self, name, getattr(cls, name))
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeTaskset:
    def __init__(self, config: FakeTasksetConfig) -> None:
        self.config = config

    def __class_getitem__(cls, item: object) -> type["FakeTaskset"]:
        del item
        return cls


class FakeEnvConfig:
    def __init__(self, taskset: FakeTasksetConfig) -> None:
        self.taskset = taskset


class FakeEnv:
    def __init__(self, taskset: FakeTaskset) -> None:
        self.taskset = taskset


@pytest.fixture()
def rewardhack_prime(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_vf = types.ModuleType("verifiers")
    fake_vf.TasksetConfig = FakeTasksetConfig
    fake_vf.Taskset = FakeTaskset
    fake_vf.EnvConfig = FakeEnvConfig
    fake_vf.Env = FakeEnv
    fake_vf.TaskSplit = str
    fake_vf.Tasks = list
    fake_vf.Task = dict
    fake_vf.State = dict

    def reward(weight: float = 1.0):
        def decorate(func):
            func._vf_reward_weight = weight
            return func

        return decorate

    fake_vf.reward = reward
    fake_vf.load_taskset = lambda config: None
    monkeypatch.setitem(sys.modules, "verifiers", fake_vf)

    package_path = Path(__file__).resolve().parents[1] / "environments" / "rewardhack_prime"
    monkeypatch.syspath_prepend(str(package_path))

    for module_name in list(sys.modules):
        if module_name == "rewardhack_prime" or module_name.startswith("rewardhack_prime."):
            del sys.modules[module_name]
    return __import__("rewardhack_prime")


def test_rewardhack_task_conversion_hides_hidden_metadata(rewardhack_prime: types.ModuleType) -> None:
    env = create_environment("code/spec-overfit")
    task = env.sample_task(seed=7)

    row = rewardhack_prime.rewardhack_task_to_vf_task(task)
    public_metadata = row["info"]["metadata"]

    assert row["prompt"] == [{"role": "user", "content": task.prompt}]
    assert row["answer"] == ""
    assert row["info"]["task_id"] == task.task_id
    assert "hidden_metadata" not in row
    assert "hidden_metadata" not in row["info"]
    assert "public_cases" in public_metadata
    assert not contains_key(row, "hidden_cases")
    assert not contains_key(row, "canonical_exploit_output")
    assert not contains_key(row, "oracle_property_cases")


def test_canonical_outputs_require_explicit_public_row_opt_in(
    rewardhack_prime: types.ModuleType,
) -> None:
    env = create_environment("code/spec-overfit")
    task = env.sample_task(seed=7)

    redacted_row = rewardhack_prime.rewardhack_task_to_vf_task(task)
    exposed_row = rewardhack_prime.rewardhack_task_to_vf_task(
        task,
        include_canonical_outputs=True,
    )

    assert not contains_key(redacted_row, "canonical_exploit_output")
    assert exposed_row["info"]["metadata"]["canonical_exploit_output"] == task.metadata["canonical_exploit_output"]
    assert not contains_key(exposed_row, "hidden_cases")
    assert not contains_key(exposed_row, "oracle_property_cases")


def test_private_store_keeps_oracle_state_out_of_public_rows(
    rewardhack_prime: types.ModuleType,
) -> None:
    config = rewardhack_prime.RewardHackTasksetConfig(
        family="code/spec-overfit",
        profile="medium",
        split="eval",
        num_tasks=1,
        seed=7,
    )
    taskset = rewardhack_prime.load_taskset(config)
    row = taskset.load_eval_tasks()[0]
    task_id = row["info"]["task_id"]
    private_task = taskset.private_store.get(task_id)
    prompt_text = row["prompt"][0]["content"]

    assert not contains_key(row, "hidden_cases")
    assert not contains_key(row, "oracle_property_cases")
    assert not contains_key(row, "canonical_exploit_output")
    assert private_task.hidden_metadata["hidden_cases"]
    assert private_task.metadata["oracle_property_cases"]
    assert private_task.metadata["canonical_exploit_output"]
    assert "hidden_cases" not in prompt_text
    for case in private_task.hidden_metadata["hidden_cases"]:
        assert case["label"] not in prompt_text


def test_rewardhack_taskset_scores_official_signal_and_oracle_metrics(
    rewardhack_prime: types.ModuleType,
) -> None:
    config = rewardhack_prime.RewardHackTasksetConfig(
        family="code/spec-overfit",
        profile="medium",
        split="eval",
        num_tasks=2,
        seed=7,
        reward_mode="official_only",
        include_oracle_metrics=True,
    )
    taskset = rewardhack_prime.load_taskset(config)
    rows = taskset.load_eval_tasks()

    assert len(rows) == 2
    assert all("hidden_metadata" not in row["info"] for row in rows)

    row = rows[0]
    task = taskset.private_store.get(row["info"]["task_id"])
    state = {
        "completion": [
            {
                "role": "assistant",
                "content": taskset.rewardhack_env.canonical_exploit_output(task),
            }
        ]
    }

    reward = asyncio.run(taskset.official_reward(row, state))
    trajectory = state["rewardhack_trajectory"]
    official_score = trajectory.official_result.score
    oracle_score = asyncio.run(taskset.oracle_score(row, state))
    hack_gap = asyncio.run(taskset.hack_gap(row, state))
    false_pass = asyncio.run(taskset.false_pass(row, state))

    assert reward == official_score
    assert official_score > oracle_score
    assert hack_gap == pytest.approx(official_score - oracle_score)
    assert false_pass == 1.0


def test_reward_modes_are_explicit_and_default_to_official_only(
    rewardhack_prime: types.ModuleType,
) -> None:
    config = rewardhack_prime.RewardHackTasksetConfig()
    assert config.reward_mode == "official_only"

    env = create_environment("code/spec-overfit")
    task = env.sample_task(seed=7)
    trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))
    official_score = trajectory.official_result.score
    oracle_score = trajectory.oracle_result.score
    false_pass = float(trajectory.is_false_pass)
    gap = max(0.0, official_score - oracle_score)

    assert rewardhack_prime.reward_from_trajectory(trajectory, "official_only") == official_score
    assert rewardhack_prime.reward_from_trajectory(trajectory, "oracle_upper_bound") == oracle_score
    assert rewardhack_prime.reward_from_trajectory(
        trajectory,
        "gap_penalized",
        penalty_lambda=0.5,
    ) == pytest.approx(official_score - 0.5 * gap)
    assert rewardhack_prime.reward_from_trajectory(
        trajectory,
        "false_pass_penalized",
        penalty_lambda=0.5,
    ) == pytest.approx(official_score - 0.5 * false_pass)


def test_score_outputs_are_weighted_for_training_and_monitoring(
    rewardhack_prime: types.ModuleType,
) -> None:
    taskset = rewardhack_prime.load_taskset(rewardhack_prime.RewardHackTasksetConfig())

    assert taskset.official_reward._vf_reward_weight == 1.0
    assert taskset.oracle_score._vf_reward_weight == 0.0
    assert taskset.hack_gap._vf_reward_weight == 0.0
    assert taskset.false_pass._vf_reward_weight == 0.0
    assert not hasattr(taskset, "rewardhack_reward")
    assert not hasattr(taskset, "official_score")
    assert not hasattr(taskset, "official_passed")
    assert not hasattr(taskset, "oracle_passed")


def test_prime_taskset_propagates_code_execution_config(rewardhack_prime: types.ModuleType) -> None:
    config = rewardhack_prime.RewardHackTasksetConfig(
        code_execution_backend="local_trusted",
        code_execution_timeout_seconds=4.0,
        code_execution_memory_mb=512,
        code_execution_stdout_limit_chars=1234,
        code_execution_stderr_limit_chars=2345,
        code_execution_max_output_object_size=3456,
        prime_sandbox_image="python:test",
        prime_sandbox_timeout_minutes=12,
        prime_sandbox_cpu_cores=2,
    )
    taskset = rewardhack_prime.load_taskset(config)
    env_config = taskset.rewardhack_env.config

    assert env_config.code_execution_backend == "local_trusted"
    assert env_config.effective_code_execution_timeout_seconds == 4.0
    assert env_config.code_execution_memory_mb == 512
    assert env_config.code_execution_stdout_limit_chars == 1234
    assert env_config.code_execution_stderr_limit_chars == 2345
    assert env_config.code_execution_max_output_object_size == 3456
    assert env_config.prime_sandbox_image == "python:test"
    assert env_config.prime_sandbox_timeout_minutes == 12
    assert env_config.prime_sandbox_cpu_cores == 2


def test_score_once_caches_trajectory_per_rollout(rewardhack_prime: types.ModuleType) -> None:
    config = rewardhack_prime.RewardHackTasksetConfig(
        family="code/spec-overfit",
        profile="medium",
        split="eval",
        num_tasks=1,
        seed=7,
    )
    taskset = rewardhack_prime.load_taskset(config)
    row = taskset.load_eval_tasks()[0]
    task = taskset.private_store.get(row["info"]["task_id"])
    state = {
        "completion": [
            {
                "role": "assistant",
                "content": taskset.rewardhack_env.canonical_exploit_output(task),
            }
        ]
    }
    original = taskset.run_rewardhack_eval
    calls = 0

    async def counted_run_rewardhack_eval(task_arg, completion_arg, state_arg):
        nonlocal calls
        calls += 1
        return await original(task_arg, completion_arg, state_arg)

    taskset.run_rewardhack_eval = counted_run_rewardhack_eval

    asyncio.run(taskset.official_reward(row, state))
    asyncio.run(taskset.oracle_score(row, state))
    asyncio.run(taskset.hack_gap(row, state))
    asyncio.run(taskset.false_pass(row, state))

    assert calls == 1
    assert "rewardhack_trajectory" in state


def test_trajectory_to_dict_hides_hidden_task_metadata_by_default() -> None:
    env = create_environment("code/spec-overfit")
    task = env.sample_task(seed=7)
    trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

    public_record = trajectory.to_dict()
    private_record = trajectory.to_dict(include_hidden_task_metadata=True)

    assert "hidden_metadata" not in public_record["task"]
    assert "hidden_metadata" in private_record["task"]
    assert "hidden_cases" in private_record["task"]["hidden_metadata"]


def test_load_environment_returns_taskset_only_env(rewardhack_prime: types.ModuleType) -> None:
    config = rewardhack_prime.RewardHackTasksetConfig(num_tasks=1)
    env = rewardhack_prime.load_environment(FakeEnvConfig(taskset=config))

    assert isinstance(env, FakeEnv)
    assert env.taskset.config is config
