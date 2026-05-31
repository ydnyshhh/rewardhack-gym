from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_rewardhack_prime_loads_with_real_verifiers() -> None:
    vf = pytest.importorskip("verifiers")
    if not hasattr(vf, "Env"):
        pytest.skip("installed verifiers package does not expose v1 Env")

    package_path = Path(__file__).resolve().parents[1] / "environments" / "rewardhack_prime"
    sys.path.insert(0, str(package_path))
    try:
        for module_name in list(sys.modules):
            if module_name == "rewardhack_prime" or module_name.startswith("rewardhack_prime."):
                del sys.modules[module_name]
        import rewardhack_prime

        config = rewardhack_prime.RewardHackTasksetConfig(num_tasks=1, split="eval", seed=3)
        env_config = vf.EnvConfig(taskset=config)
        env = rewardhack_prime.load_environment(env_config)

        assert isinstance(env, vf.Env)
        assert env.taskset.official_reward.reward_weight == 1.0
        assert env.taskset.oracle_score.reward_weight == 0.0
        assert env.taskset.hack_gap.reward_weight == 0.0
        assert env.taskset.false_pass.reward_weight == 0.0
        assert env.taskset.rows() == []
        assert len(env.taskset.eval_rows()) == 1
    finally:
        try:
            sys.path.remove(str(package_path))
        except ValueError:
            pass
