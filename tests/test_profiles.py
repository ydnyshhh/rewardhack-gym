from rewardhack_gym import create_environment
from rewardhack_gym.core.config import EnvironmentConfig, ExploitabilityProfile


def test_exploitability_profile_presets_scale_as_expected() -> None:
    aligned = ExploitabilityProfile.from_level("aligned")
    low = ExploitabilityProfile.from_level("low")
    medium = ExploitabilityProfile.from_level("medium")
    adversarial = ExploitabilityProfile.from_level("adversarial")

    assert aligned.official_coverage_fraction == 0.95
    assert aligned.hidden_adversarial_strength == 0.2
    assert aligned.parser_strictness == 0.95
    assert aligned.process_validation_fraction == 0.9
    assert aligned.domain_awareness == 0.9
    assert aligned.perturbation_robustness == 0.2
    assert aligned.public_example_diversity == 0.95
    assert aligned.official_coverage_fraction > low.official_coverage_fraction
    assert low.official_coverage_fraction > medium.official_coverage_fraction > adversarial.official_coverage_fraction
    assert aligned.domain_awareness > low.domain_awareness
    assert low.domain_awareness > medium.domain_awareness > adversarial.domain_awareness


def test_profile_overrides_produce_new_profile() -> None:
    medium = ExploitabilityProfile.from_level("medium")
    updated = medium.with_overrides(domain_awareness=0.9)

    assert medium.domain_awareness == 0.3
    assert updated.domain_awareness == 0.9


def test_environment_config_carries_code_execution_settings() -> None:
    config = EnvironmentConfig.from_profile(
        profile="aligned",
        code_execution_backend="prime_sandbox",
        code_execution_timeout_seconds=3.5,
        code_execution_memory_mb=512,
        code_execution_stdout_limit_chars=1234,
        code_execution_stderr_limit_chars=2345,
        code_execution_max_output_object_size=3456,
        prime_sandbox_image="python:test",
        prime_sandbox_timeout_minutes=12,
        prime_sandbox_cpu_cores=2,
    )

    assert config.code_execution_backend == "prime_sandbox"
    assert config.effective_code_execution_timeout_seconds == 3.5
    assert config.code_execution_memory_mb == 512
    assert config.code_execution_stdout_limit_chars == 1234
    assert config.code_execution_stderr_limit_chars == 2345
    assert config.code_execution_max_output_object_size == 3456
    assert config.prime_sandbox_image == "python:test"
    assert config.prime_sandbox_timeout_minutes == 12
    assert config.prime_sandbox_cpu_cores == 2


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


def test_profiles_causally_change_schema_preservation_official_strength() -> None:
    low_env = create_environment(
        "code/schema-preservation",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("low")),
    )
    medium_env = create_environment(
        "code/schema-preservation",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("medium")),
    )

    low_task = low_env.sample_task(seed=5)
    medium_task = medium_env.sample_task(seed=5)
    low_exploit = low_env.evaluate_output(low_task, low_env.canonical_exploit_output(low_task))
    medium_exploit = medium_env.evaluate_output(medium_task, medium_env.canonical_exploit_output(medium_task))

    assert not low_exploit.official_result.passed
    assert medium_exploit.official_result.passed


def test_profiles_change_schema_preservation_structure_not_just_scores() -> None:
    low_env = create_environment(
        "code/schema-preservation",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("low")),
    )
    medium_env = create_environment(
        "code/schema-preservation",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("medium")),
    )

    low_task = low_env.sample_task(seed=5)
    medium_task = medium_env.sample_task(seed=5)
    low_structure = low_task.metadata["structural_profile"]
    medium_structure = medium_task.metadata["structural_profile"]

    assert low_structure["official_checks_order_stability"] != medium_structure["official_checks_order_stability"]
    assert low_structure["official_checks_nested_aliasing"] != medium_structure["official_checks_nested_aliasing"]


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


def test_profiles_change_reasoning_validity_structure_not_just_scores() -> None:
    low_env = create_environment(
        "math/reasoning-validity",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("low")),
    )
    medium_env = create_environment(
        "math/reasoning-validity",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("medium")),
    )

    low_task = low_env.sample_task(seed=5)
    medium_task = medium_env.sample_task(seed=5)
    low_structure = low_task.metadata["structural_profile"]
    medium_structure = medium_task.metadata["structural_profile"]

    assert low_structure["official_support_mode"] != medium_structure["official_support_mode"]


def test_profiles_causally_change_piecewise_reasoning_official_strength() -> None:
    low_env = create_environment(
        "math/piecewise-reasoning",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("low")),
    )
    medium_env = create_environment(
        "math/piecewise-reasoning",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("medium")),
    )

    low_task = low_env.sample_task(seed=5)
    medium_task = medium_env.sample_task(seed=5)
    low_exploit = low_env.evaluate_output(low_task, low_env.canonical_exploit_output(low_task))
    medium_exploit = medium_env.evaluate_output(medium_task, medium_env.canonical_exploit_output(medium_task))

    assert not low_exploit.official_result.passed
    assert medium_exploit.official_result.passed


def test_profiles_change_piecewise_reasoning_structure_not_just_scores() -> None:
    low_env = create_environment(
        "math/piecewise-reasoning",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("low")),
    )
    medium_env = create_environment(
        "math/piecewise-reasoning",
        EnvironmentConfig(exploitability=ExploitabilityProfile.from_level("medium")),
    )

    low_task = low_env.sample_task(seed=5)
    medium_task = medium_env.sample_task(seed=5)
    low_structure = low_task.metadata["structural_profile"]
    medium_structure = medium_task.metadata["structural_profile"]

    assert low_structure["official_support_mode"] != medium_structure["official_support_mode"]
