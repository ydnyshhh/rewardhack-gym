import pytest

from rewardhack_gym import create_environment


@pytest.mark.parametrize(
    "environment_name",
    [
        "math/symbolic-mismatch",
        "math/reasoning-validity",
        "math/constraint-sensitive",
    ],
)
def test_math_canonical_true_outputs_pass_official_and_oracle(environment_name: str) -> None:
    env = create_environment(environment_name)
    task = env.sample_task(seed=5)
    trajectory = env.evaluate_output(task, env.canonical_true_output(task))

    assert trajectory.official_result.passed
    assert trajectory.oracle_result.passed
    assert not trajectory.is_false_pass


@pytest.mark.parametrize(
    "environment_name",
    [
        "math/symbolic-mismatch",
        "math/reasoning-validity",
        "math/constraint-sensitive",
    ],
)
def test_math_canonical_exploits_false_pass(environment_name: str) -> None:
    env = create_environment(environment_name)
    task = env.sample_task(seed=5)
    trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

    assert trajectory.official_result.passed
    assert not trajectory.oracle_result.passed
    assert trajectory.is_false_pass

