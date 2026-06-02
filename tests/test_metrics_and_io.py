import json
from pathlib import Path

import pytest

from rewardhack_gym import create_environment
from rewardhack_gym.analysis import summarize_best_of_n, summarize_trajectories
from rewardhack_gym.io import read_jsonl, write_jsonl
from rewardhack_gym.core.models import CheckerResult, EvaluatorResult, ExploitSurface, RuntimeMetadata, Task, Trajectory
from rewardhack_gym.runners.selection import BestOfNResult, BestOfNRunner


def test_metrics_and_jsonl_roundtrip() -> None:
    env = create_environment("code/spec-overfit")
    task = env.sample_task(seed=7)
    trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

    summary = summarize_trajectories([trajectory])
    assert summary.false_pass_rate == 1.0
    assert summary.hack_rate == 1.0
    assert summary.official_pass_rate == 1.0
    assert summary.oracle_pass_rate == 0.0
    assert summary.true_pass_rate == 0.0
    assert summary.official_oracle_disagreement_rate == 1.0
    assert summary.mean_hack_gap > 0.0
    assert summary.conditional_oracle_score_given_official_pass == trajectory.oracle_result.score
    assert "hack_rate" in summary.confidence_intervals

    output_dir = Path("tests_artifacts")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "metrics_roundtrip.jsonl"
    write_jsonl(output_path, [trajectory])
    records = read_jsonl(output_path)
    assert len(records) == 1
    assert records[0]["is_false_pass"] is True
    json.dumps(records[0])


def make_trajectory(
    *,
    official_score: float,
    official_passed: bool,
    oracle_score: float,
    oracle_passed: bool,
    profile: str,
) -> Trajectory:
    task = Task(
        task_id=f"task-{profile}-{official_score}-{oracle_score}",
        family="synthetic",
        prompt="prompt",
        expected_interface="text",
        difficulty="medium",
        exploit_surface=ExploitSurface(
            summary="synthetic",
            ignored_dimensions=(),
            exploit_families=("synthetic-hack",),
            cheaper_path_hypotheses=(),
        ),
        metadata={"profile": profile},
    )
    return Trajectory(
        task=task,
        prompt=task.prompt,
        final_output="output",
        official_result=EvaluatorResult(
            evaluator_name="official",
            score=official_score,
            passed=official_passed,
            components=(CheckerResult("official", official_score, official_passed),),
        ),
        oracle_result=EvaluatorResult(
            evaluator_name="oracle",
            score=oracle_score,
            passed=oracle_passed,
            components=(CheckerResult("oracle", oracle_score, oracle_passed),),
        ),
        exploit_labels=("synthetic-hack",) if official_passed and not oracle_passed else (),
        runtime=RuntimeMetadata(environment_profile=profile),
    )


def test_research_metrics_include_disagreement_profile_sensitivity_and_cis() -> None:
    trajectories = [
        make_trajectory(official_score=1.0, official_passed=True, oracle_score=1.0, oracle_passed=True, profile="low"),
        make_trajectory(official_score=1.0, official_passed=True, oracle_score=0.2, oracle_passed=False, profile="medium"),
        make_trajectory(official_score=0.2, official_passed=False, oracle_score=1.0, oracle_passed=True, profile="high"),
        make_trajectory(official_score=1.0, official_passed=True, oracle_score=0.0, oracle_passed=False, profile="high"),
    ]

    summary = summarize_trajectories(trajectories, bootstrap_samples=100, bootstrap_seed=3)
    data = summary.to_dict()

    assert summary.official_pass_rate == 0.75
    assert summary.oracle_pass_rate == 0.5
    assert summary.true_pass_rate == 0.25
    assert summary.false_pass_rate == 0.5
    assert summary.hack_rate == 0.5
    assert summary.false_fail_rate == 0.25
    assert summary.official_oracle_disagreement_rate == 0.75
    assert summary.mean_hack_gap == 0.45
    assert summary.mean_hack_gap_given_official_pass == 0.6
    assert summary.conditional_oracle_score_given_official_pass == pytest.approx(0.4)
    assert summary.hack_rate_by_profile == {"high": 0.5, "low": 0.0, "medium": 1.0}
    assert summary.hack_gap_by_profile == {"high": 0.5, "low": 0.0, "medium": 0.8}
    assert summary.exploit_sensitivity_slope > 0.0
    assert data["confidence_intervals"]["hack_rate"]["low"] <= summary.hack_rate
    assert data["confidence_intervals"]["oracle_pass_rate"]["high"] >= summary.oracle_pass_rate
    assert data["confidence_intervals"]["hack_gap"]["mean"] == summary.mean_hack_gap


def test_best_of_n_metrics_and_selection_mode_metadata() -> None:
    first = make_trajectory(
        official_score=0.9,
        official_passed=True,
        oracle_score=0.9,
        oracle_passed=True,
        profile="medium",
    )
    selected_hack = make_trajectory(
        official_score=1.0,
        official_passed=True,
        oracle_score=0.1,
        oracle_passed=False,
        profile="medium",
    )
    selected_clean = make_trajectory(
        official_score=0.95,
        official_passed=True,
        oracle_score=0.95,
        oracle_passed=True,
        profile="medium",
    )
    result = summarize_best_of_n(
        [
            BestOfNResult(
                selected=selected_hack,
                candidates=(first, selected_hack),
                selected_index=1,
                selection_mode="official_then_low_oracle",
            ),
            BestOfNResult(
                selected=selected_clean,
                candidates=(selected_clean, selected_hack),
                selected_index=0,
                selection_mode="official_then_high_oracle",
            ),
        ]
    )
    data = result.to_dict()

    assert data["official_pass@2"] == 1.0
    assert data["oracle_pass@2"] == 0.5
    assert data["hack@2"] == 0.5
    assert data["true_pass@2"] == 0.5
    assert data["hack@1"] == 0.0
    assert data["best_of_n_hack_amplification"] == 0.5
    assert data["selection_mode_counts"] == {
        "official_then_high_oracle": 1,
        "official_then_low_oracle": 1,
    }


def test_best_of_n_selection_modes_make_oracle_tie_breaks_explicit() -> None:
    low_oracle_hack = make_trajectory(
        official_score=1.0,
        official_passed=True,
        oracle_score=0.1,
        oracle_passed=False,
        profile="medium",
    )
    high_oracle_clean = make_trajectory(
        official_score=1.0,
        official_passed=True,
        oracle_score=0.9,
        oracle_passed=True,
        profile="medium",
    )
    oracle_upper_bound = make_trajectory(
        official_score=0.5,
        official_passed=False,
        oracle_score=1.0,
        oracle_passed=True,
        profile="medium",
    )
    candidates = (low_oracle_hack, high_oracle_clean, oracle_upper_bound)

    assert BestOfNRunner._select_index(
        candidates,
        selection_mode="official_only",
        selection_seed=0,
    ) == 0
    assert BestOfNRunner._select_index(
        candidates,
        selection_mode="official_then_low_oracle",
        selection_seed=0,
    ) == 0
    assert BestOfNRunner._select_index(
        candidates,
        selection_mode="official_then_high_oracle",
        selection_seed=0,
    ) == 1
    assert BestOfNRunner._select_index(
        candidates,
        selection_mode="oracle_upper_bound",
        selection_seed=0,
    ) == 2


def test_best_of_n_runner_defaults_to_official_only_selection() -> None:
    candidates = (
        make_trajectory(official_score=1.0, official_passed=True, oracle_score=0.1, oracle_passed=False, profile="medium"),
        make_trajectory(official_score=1.0, official_passed=True, oracle_score=0.9, oracle_passed=True, profile="medium"),
    )

    class FakeEnvironment:
        def evaluate_output(self, task, output, *, policy_id=None, annotations=None):
            del task, policy_id, annotations
            return candidates[int(output)]

    runner = BestOfNRunner(FakeEnvironment())
    result = runner.run(candidates[0].task, lambda task, index: str(index), n=2)

    assert result.selection_mode == "official_only"
    assert result.selected_index == 0
