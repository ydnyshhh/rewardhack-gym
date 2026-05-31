from __future__ import annotations

import verifiers as vf

from rewardhack_prime.config import RewardHackTasksetConfig
from rewardhack_prime.taskset import load_taskset


def _load_taskset(taskset_config: object) -> vf.Taskset:
    if isinstance(taskset_config, RewardHackTasksetConfig):
        return load_taskset(taskset_config)

    verifiers_loader = getattr(vf, "load_taskset", None)
    if verifiers_loader is not None:
        return verifiers_loader(config=taskset_config)

    if isinstance(taskset_config, vf.TasksetConfig):
        return vf.Taskset(config=taskset_config)

    raise TypeError("RewardHack load_environment expected a RewardHackTasksetConfig or Verifiers TasksetConfig.")


def _load_harness(config: vf.EnvConfig) -> object | None:
    harness_config = getattr(config, "harness", None)
    if harness_config is None:
        return None

    verifiers_loader = getattr(vf, "load_harness", None)
    if verifiers_loader is not None:
        return verifiers_loader(config=harness_config)

    harness_cls = getattr(vf, "Harness", None)
    if harness_cls is not None:
        return harness_cls(config=harness_config)

    return None


def load_environment(config: vf.EnvConfig) -> vf.Env:
    taskset = _load_taskset(config.taskset)
    return vf.Env(taskset=taskset, harness=_load_harness(config))
