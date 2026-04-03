from rewardhack_gym import create_environment
from rewardhack_gym.core.config import EnvironmentConfig, ExploitabilityProfile


def test_exploitability_profile_presets_scale_as_expected() -> None:
    low = ExploitabilityProfile.from_level("low")
    medium = ExploitabilityProfile.from_level("medium")
    adversarial = ExploitabilityProfile.from_level("adversarial")

    assert low.official_coverage_fraction > medium.official_coverage_fraction > adversarial.official_coverage_fraction
    assert low.domain_awareness > medium.domain_awareness > adversarial.domain_awareness


def test_profile_overrides_produce_new_profile() -> None:
    medium = ExploitabilityProfile.from_level("medium")
    updated = medium.with_overrides(domain_awareness=0.9)

    assert medium.domain_awareness == 0.3
    assert updated.domain_awareness == 0.9


def test_profiles_causally_change_code_spec_overfit_official_strength() -> None:
    low_env = create_environment(
        "code/spec-overfit",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("low")),
    )
    medium_env = create_environment(
        "code/spec-overfit",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("medium")),
    )

    low_task = low_env.sample_task(seed=5)
    medium_task = medium_env.sample_task(seed=5)
    low_exploit = low_env.evaluate_output(low_task, low_env.canonical_exploit_output(low_task))
    medium_exploit = medium_env.evaluate_output(medium_task, medium_env.canonical_exploit_output(medium_task))

    assert not low_exploit.official_result.passed
    assert medium_exploit.official_result.passed
    assert low_exploit.official_result.score < medium_exploit.official_result.score


def test_profiles_causally_change_math_symbolic_official_strength() -> None:
    low_env = create_environment(
        "math/symbolic-mismatch",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("low")),
    )
    medium_env = create_environment(
        "math/symbolic-mismatch",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("medium")),
    )

    low_task = low_env.sample_task(seed=5)
    medium_task = medium_env.sample_task(seed=5)
    low_exploit = low_env.evaluate_output(low_task, low_env.canonical_exploit_output(low_task))
    medium_exploit = medium_env.evaluate_output(medium_task, medium_env.canonical_exploit_output(medium_task))

    assert not low_exploit.official_result.passed
    assert medium_exploit.official_result.passed
    assert low_exploit.official_result.score < medium_exploit.official_result.score


def test_profiles_causally_change_patch_verification_official_strength() -> None:
    low_env = create_environment(
        "code/patch-verification",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("low")),
    )
    medium_env = create_environment(
        "code/patch-verification",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("medium")),
    )

    low_task = low_env.sample_task(seed=5)
    medium_task = medium_env.sample_task(seed=5)
    low_exploit = low_env.evaluate_output(low_task, low_env.canonical_exploit_output(low_task))
    medium_exploit = medium_env.evaluate_output(medium_task, medium_env.canonical_exploit_output(medium_task))

    assert not low_exploit.official_result.passed
    assert medium_exploit.official_result.passed


def test_profiles_change_patch_verification_structure_not_just_scores() -> None:
    low_env = create_environment(
        "code/patch-verification",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("low")),
    )
    medium_env = create_environment(
        "code/patch-verification",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("medium")),
    )

    low_task = low_env.sample_task(seed=5)
    medium_task = medium_env.sample_task(seed=5)
    low_structure = low_task.metadata["structural_profile"]
    medium_structure = medium_task.metadata["structural_profile"]

    assert low_structure["official_exposes_tie_resolution"] != medium_structure["official_exposes_tie_resolution"]
    assert low_structure["official_side_effect_checks"] != medium_structure["official_side_effect_checks"]


def test_profiles_causally_change_constraint_sensitive_official_strength() -> None:
    low_env = create_environment(
        "math/constraint-sensitive",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("low")),
    )
    medium_env = create_environment(
        "math/constraint-sensitive",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("medium")),
    )

    low_task = low_env.sample_task(seed=5)
    medium_task = medium_env.sample_task(seed=5)
    low_exploit = low_env.evaluate_output(low_task, low_env.canonical_exploit_output(low_task))
    medium_exploit = medium_env.evaluate_output(medium_task, medium_env.canonical_exploit_output(medium_task))

    assert not low_exploit.official_result.passed
    assert medium_exploit.official_result.passed


def test_profiles_change_constraint_sensitive_structure_not_just_scores() -> None:
    low_env = create_environment(
        "math/constraint-sensitive",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("low")),
    )
    medium_env = create_environment(
        "math/constraint-sensitive",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("medium")),
    )

    low_task = low_env.sample_task(seed=5)
    medium_task = medium_env.sample_task(seed=5)
    low_structure = low_task.metadata["structural_profile"]
    medium_structure = medium_task.metadata["structural_profile"]

    assert low_structure["official_probe_regime"] != medium_structure["official_probe_regime"]
    assert low_structure["official_domain_mode"] != medium_structure["official_domain_mode"]
