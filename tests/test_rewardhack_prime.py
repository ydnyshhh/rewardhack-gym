from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from rewardhack_gym import create_environment


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

    assert row["prompt"] == [{"role": "user", "content": task.prompt}]
    assert row["answer"] == ""
    assert row["info"]["task_id"] == task.task_id
    assert "hidden_metadata" not in row
    assert "hidden_metadata" not in row["info"]


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
    task = taskset._tasks_by_id[row["info"]["task_id"]]
    state = {
        "completion": [
            {
                "role": "assistant",
                "content": taskset.rewardhack_env.canonical_exploit_output(task),
            }
        ]
    }

    reward = asyncio.run(taskset.rewardhack_reward(row, state))
    official_score = asyncio.run(taskset.official_score(row, state))
    oracle_score = asyncio.run(taskset.oracle_score(row, state))
    hack_gap = asyncio.run(taskset.hack_gap(row, state))
    false_pass = asyncio.run(taskset.false_pass(row, state))

    assert reward == official_score
    assert official_score > oracle_score
    assert hack_gap == pytest.approx(official_score - oracle_score)
    assert false_pass == 1.0


def test_load_environment_returns_taskset_only_env(rewardhack_prime: types.ModuleType) -> None:
    config = rewardhack_prime.RewardHackTasksetConfig(num_tasks=1)
    env = rewardhack_prime.load_environment(FakeEnvConfig(taskset=config))

    assert isinstance(env, FakeEnv)
    assert env.taskset.config is config
