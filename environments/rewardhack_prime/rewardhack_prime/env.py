from __future__ import annotations

import verifiers as vf

from rewardhack_prime.config import RewardHackTasksetConfig
from rewardhack_prime.taskset import load_taskset


def load_environment(config: vf.EnvConfig) -> vf.Env:
    taskset_config = config.taskset
    if not isinstance(taskset_config, RewardHackTasksetConfig):
        taskset = vf.load_taskset(config=taskset_config)
    else:
        taskset = load_taskset(taskset_config)
    return vf.Env(taskset=taskset)
