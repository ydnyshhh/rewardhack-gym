from pathlib import Path

from rewardhack_gym import create_environment
from rewardhack_gym.analysis import build_matched_pairs, build_mech_interp_record, build_mech_interp_records
from rewardhack_gym.io import read_jsonl, write_jsonl


def test_mech_interp_export_contains_core_fields_and_stable_ids() -> None:
    env = create_environment("code/patch-verification")
    task = env.sample_task(seed=4)
    trajectory = env.evaluate_output(task, env.canonical_exploit_output(task))

    exported = build_mech_interp_record(trajectory)
    exported_again = build_mech_interp_record(trajectory)

    assert exported.environment_name == "code/patch-verification"
    assert exported.family == "code"
    assert exported.scenario_id == str(task.metadata["scenario_id"])
    assert exported.exploit_class == str(task.metadata["exploit_mode"])
    assert exported.semantic_failures
    assert exported.outcome_label == "false-pass"
    assert exported.canonical_output_type == "canonical_exploit"
    assert exported.trace_id == exported_again.trace_id
    assert exported.scenario_cohort_id == exported_again.scenario_cohort_id
    assert exported.failure_slice_id == exported_again.failure_slice_id


def test_matched_pairs_prefer_exact_task_matches() -> None:
    env = create_environment("math/reasoning-validity")
    task = env.sample_task(seed=5)
    true_pass = env.evaluate_output(task, env.canonical_true_output(task))
    false_pass = env.evaluate_output(task, env.canonical_exploit_output(task))

    pairs = build_matched_pairs([true_pass, false_pass])

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["match_level"] == "exact-task"
    assert pair["environment_name"] == "math/reasoning-validity"
    assert pair["true_pass"]["outcome_label"] == "true-pass"
    assert pair["false_pass"]["outcome_label"] == "false-pass"
    assert pair["true_pass"]["scenario_cohort_id"] == pair["false_pass"]["scenario_cohort_id"]


def test_mech_interp_export_roundtrip_and_pair_building_from_jsonl() -> None:
    env = create_environment("code/schema-preservation")
    task = env.sample_task(seed=7)
    records = [
        env.evaluate_output(task, env.canonical_true_output(task)),
        env.evaluate_output(task, env.canonical_exploit_output(task)),
    ]

    artifact_dir = Path("tests_artifacts")
    artifact_dir.mkdir(exist_ok=True)
    traces_path = artifact_dir / "mech_interp_traces.jsonl"
    exported_path = artifact_dir / "mech_interp_rows.jsonl"
    write_jsonl(traces_path, records)

    exported = build_mech_interp_records(read_jsonl(traces_path))
    write_jsonl(exported_path, exported)
    roundtripped = read_jsonl(exported_path)
    pairs = build_matched_pairs(roundtripped)

    assert len(roundtripped) == 2
    assert {row["canonical_output_type"] for row in roundtripped} == {"canonical_true", "canonical_exploit"}
    assert len(pairs) == 1
    assert pairs[0]["false_pass"]["family"] == "code"
