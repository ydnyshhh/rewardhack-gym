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


def test_constraint_sensitive_family_generates_multiple_scenarios_and_exploits() -> None:
    env = create_environment("math/constraint-sensitive")
    tasks = [env.sample_task(seed=seed) for seed in range(1, 10)]

    scenario_ids = {str(task.metadata["scenario_id"]) for task in tasks}
    exploit_modes = {str(task.metadata["exploit_mode"]) for task in tasks}

    assert len(scenario_ids) >= 2
    assert len(exploit_modes) >= 2


def test_constraint_sensitive_canonical_exploit_false_passes_across_multiple_seeds() -> None:
    env = create_environment("math/constraint-sensitive")

    for seed in (1, 4, 7):
        task = env.sample_task(seed=seed)
        trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

        assert trajectory.official_result.passed
        assert not trajectory.oracle_result.passed


def test_constraint_sensitive_annotations_capture_failure_structure() -> None:
    env = create_environment("math/constraint-sensitive")
    task = env.sample_task(seed=4)
    trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

    assert trajectory.annotations["scenario_id"] == task.metadata["scenario_id"]
    assert trajectory.annotations["canonical_exploit_class"] == task.metadata["exploit_mode"]
    assert trajectory.annotations["visible_probe_points"]
    assert trajectory.annotations["hidden_probe_points"]
    assert trajectory.annotations["semantic_failures"]
