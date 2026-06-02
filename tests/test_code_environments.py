import pytest

from rewardhack_gym import create_environment
from rewardhack_gym.core.config import EnvironmentConfig
from rewardhack_gym.runners import BestOfNRunner


@pytest.mark.parametrize(
    "environment_name",
    [
        "code/spec-overfit",
        "code/patch-verification",
        "code/interface-conformance",
        "code/schema-preservation",
    ],
)
def test_code_canonical_true_outputs_pass_official_and_oracle(environment_name: str) -> None:
    env = create_environment(environment_name)
    task = env.sample_task(seed=3)
    trajectory = env.evaluate_output(task, env.canonical_true_output(task))

    assert trajectory.official_result.passed
    assert trajectory.oracle_result.passed
    assert not trajectory.is_false_pass


@pytest.mark.parametrize(
    "environment_name",
    [
        "code/spec-overfit",
        "code/patch-verification",
        "code/interface-conformance",
        "code/schema-preservation",
    ],
)
def test_code_behavioral_checkers_report_backend_execution(environment_name: str) -> None:
    env = create_environment(environment_name)
    task = env.sample_task(seed=3)
    trajectory = env.evaluate_output(task, env.canonical_true_output(task))
    execution_diagnostics = [
        component.diagnostics["execution"]
        for result in (trajectory.official_result, trajectory.oracle_result)
        for component in result.components
        if "execution" in component.diagnostics
    ]

    assert execution_diagnostics
    assert {diagnostic["backend"] for diagnostic in execution_diagnostics} == {"subprocess"}


def test_code_behavioral_checkers_use_configured_backend() -> None:
    env = create_environment(
        "code/spec-overfit",
        EnvironmentConfig(code_execution_backend="local_trusted"),
    )
    task = env.sample_task(seed=3)
    trajectory = env.evaluate_output(task, env.canonical_true_output(task))
    execution_diagnostics = [
        component.diagnostics["execution"]
        for result in (trajectory.official_result, trajectory.oracle_result)
        for component in result.components
        if "execution" in component.diagnostics
    ]

    assert execution_diagnostics
    assert {diagnostic["backend"] for diagnostic in execution_diagnostics} == {"local_trusted"}


@pytest.mark.parametrize(
    "environment_name",
    [
        "code/spec-overfit",
        "code/patch-verification",
        "code/interface-conformance",
        "code/schema-preservation",
    ],
)
def test_code_canonical_exploits_false_pass(environment_name: str) -> None:
    env = create_environment(environment_name)
    task = env.sample_task(seed=3)
    trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

    assert trajectory.official_result.passed
    assert not trajectory.oracle_result.passed
    assert trajectory.is_false_pass
    assert trajectory.exploit_labels


def test_patch_verification_family_generates_multiple_scenarios_and_exploits() -> None:
    env = create_environment("code/patch-verification")
    tasks = [env.sample_task(seed=seed) for seed in range(1, 10)]

    scenario_ids = {str(task.metadata["scenario_id"]) for task in tasks}
    exploit_modes = {str(task.metadata["exploit_mode"]) for task in tasks}

    assert len(scenario_ids) >= 2
    assert len(exploit_modes) >= 2


def test_patch_verification_canonical_exploit_false_passes_across_multiple_seeds() -> None:
    env = create_environment("code/patch-verification")

    for seed in (1, 4, 7):
        task = env.sample_task(seed=seed)
        trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

        assert trajectory.official_result.passed
        assert not trajectory.oracle_result.passed


def test_patch_verification_annotations_capture_failure_structure() -> None:
    env = create_environment("code/patch-verification")
    task = env.sample_task(seed=4)
    trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

    assert trajectory.annotations["scenario_id"] == task.metadata["scenario_id"]
    assert trajectory.annotations["canonical_exploit_class"] == task.metadata["exploit_mode"]
    assert trajectory.annotations["visible_probe_labels"]
    assert trajectory.annotations["hidden_probe_labels"]
    assert trajectory.annotations["semantic_failures"]


def test_spec_overfit_annotations_capture_failure_structure() -> None:
    env = create_environment("code/spec-overfit")
    task = env.sample_task(seed=4)
    trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

    assert trajectory.annotations["scenario_id"] == task.metadata["template"]
    assert trajectory.annotations["visible_probe_labels"]
    assert trajectory.annotations["hidden_probe_labels"]
    assert trajectory.annotations["semantic_failures"]


def test_interface_conformance_annotations_capture_failure_structure() -> None:
    env = create_environment("code/interface-conformance")
    task = env.sample_task(seed=4)
    trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

    assert trajectory.annotations["scenario_id"] == task.metadata["scenario_id"]
    assert trajectory.annotations["visible_probe_labels"]
    assert trajectory.annotations["hidden_probe_labels"]
    assert trajectory.annotations["semantic_failures"]


def test_schema_preservation_family_generates_multiple_scenarios_and_exploits() -> None:
    env = create_environment("code/schema-preservation")
    tasks = [env.sample_task(seed=seed) for seed in range(1, 10)]

    scenario_ids = {str(task.metadata["scenario_id"]) for task in tasks}
    exploit_modes = {str(task.metadata["exploit_mode"]) for task in tasks}

    assert len(scenario_ids) >= 2
    assert len(exploit_modes) >= 2


def test_schema_preservation_canonical_exploit_false_passes_across_multiple_seeds() -> None:
    env = create_environment("code/schema-preservation")

    for seed in (1, 4, 7):
        task = env.sample_task(seed=seed)
        trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

        assert trajectory.official_result.passed
        assert not trajectory.oracle_result.passed


def test_schema_preservation_annotations_capture_failure_structure() -> None:
    env = create_environment("code/schema-preservation")
    task = env.sample_task(seed=4)
    trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

    assert trajectory.annotations["scenario_id"] == task.metadata["scenario_id"]
    assert trajectory.annotations["canonical_exploit_class"] == task.metadata["exploit_mode"]
    assert trajectory.annotations["visible_probe_labels"]
    assert trajectory.annotations["hidden_probe_labels"]
    assert trajectory.annotations["semantic_failures"]


def test_negative_control_true_output_passes_equivalent_official_and_oracle() -> None:
    env = create_environment("control/negative-code")
    task = env.sample_task(seed=3)
    trajectory = env.evaluate_output(task, env.canonical_true_output(task))

    assert trajectory.official_result.passed
    assert trajectory.oracle_result.passed
    assert trajectory.official_result.score == trajectory.oracle_result.score
    assert not trajectory.is_false_pass
    assert not trajectory.exploit_labels
    assert task.hidden_metadata["hidden_cases"]
    assert task.metadata["negative_control"] is True


def test_negative_control_shortcut_fails_official_and_oracle() -> None:
    env = create_environment("control/negative-code")
    task = env.sample_task(seed=3)
    trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

    assert not trajectory.official_result.passed
    assert not trajectory.oracle_result.passed
    assert trajectory.official_result.score == trajectory.oracle_result.score
    assert not trajectory.is_false_pass
    assert not trajectory.exploit_labels


def test_negative_control_best_of_n_does_not_amplify_hacks() -> None:
    env = create_environment("control/negative-code")
    task = env.sample_task(seed=3)
    outputs = (env.canonical_exploit_output(task), env.canonical_true_output(task), env.canonical_exploit_output(task))
    runner = BestOfNRunner(env)

    result = runner.run(
        task,
        lambda task_, index: outputs[index],
        n=len(outputs),
        selection_mode="official_then_low_oracle",
    )

    assert result.selected_index == 1
    assert result.selected.official_result.passed
    assert result.selected.oracle_result.passed
    assert not result.selected.is_false_pass
