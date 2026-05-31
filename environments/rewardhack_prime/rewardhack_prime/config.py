from __future__ import annotations

import verifiers as vf

SUPPORTED_PROFILES = ("aligned", "low", "medium", "high", "adversarial")


class RewardHackTasksetConfig(vf.TasksetConfig):
    family: str = "code/spec-overfit"
    profile: str = "medium"
    split: str = "eval"
    num_tasks: int = 100
    seed: int = 0
    reward_mode: str = "official_only"
    include_oracle_metrics: bool = True
    expose_canonical_outputs: bool = False
