from __future__ import annotations

import pytest

from rewardhack_gym.experiments.metrics import compute_candidate_metrics


def record(
    *,
    official_score: float,
    official_passed: bool,
    oracle_score: float,
    oracle_passed: bool,
    profile: str = "medium",
) -> dict:
    if official_passed and oracle_passed:
        label = "true_pass"
    elif official_passed and not oracle_passed:
        label = "false_pass"
    elif not official_passed and oracle_passed:
        label = "false_fail"
    else:
        label = "true_fail"
    return {
        "official_score": official_score,
        "official_passed": official_passed,
        "oracle_score": oracle_score,
        "oracle_passed": oracle_passed,
        "hack_gap": official_score - oracle_score,
        "false_pass": label == "false_pass",
        "outcome_label": label,
        "profile": profile,
    }


def test_experiment_metrics_core_rates_and_gaps() -> None:
    summary = compute_candidate_metrics(
        [
            record(official_score=1.0, official_passed=True, oracle_score=1.0, oracle_passed=True, profile="low"),
            record(official_score=1.0, official_passed=True, oracle_score=0.2, oracle_passed=False, profile="medium"),
            record(official_score=0.2, official_passed=False, oracle_score=1.0, oracle_passed=True, profile="high"),
            record(official_score=0.0, official_passed=False, oracle_score=0.0, oracle_passed=False, profile="high"),
        ],
        bootstrap=True,
        bootstrap_samples=50,
    )

    assert summary.official_pass_rate == 0.5
    assert summary.oracle_pass_rate == 0.5
    assert summary.false_pass_rate == 0.25
    assert summary.false_fail_rate == 0.25
    assert summary.true_pass_rate == 0.25
    assert summary.true_fail_rate == 0.25
    assert summary.mean_hack_gap == pytest.approx(0.0)
    assert summary.mean_positive_hack_gap == pytest.approx(0.2)
    assert summary.mean_hack_gap_given_official_pass == pytest.approx(0.4)
    assert summary.conditional_oracle_score_given_official_pass == pytest.approx(0.6)
    assert summary.disagreement_rate == 0.5
    assert "false_pass_rate" in summary.confidence_intervals

