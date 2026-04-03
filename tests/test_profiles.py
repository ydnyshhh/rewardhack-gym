from rewardhack_gym.core.config import ExploitabilityProfile


def test_exploitability_profile_presets_scale_as_expected() -> None:
    low = ExploitabilityProfile.from_level("low")
    medium = ExploitabilityProfile.from_level("medium")
    adversarial = ExploitabilityProfile.from_level("adversarial")

    assert low.official_coverage_fraction > medium.official_coverage_fraction > adversarial.official_coverage_fraction
    assert low.domain_awareness > medium.domain_awareness > adversarial.domain_awareness

