from __future__ import annotations

from rewardhack_gym import EnvironmentConfig, create_environment


def test_task_ids_are_stable_content_addressed_and_versioned() -> None:
    config = EnvironmentConfig.from_profile(profile="high", seed=0)
    env = create_environment("code/spec-overfit", config)

    first = env.sample_task(seed=7)
    second = env.sample_task(seed=7)

    assert first.task_id == second.task_id
    assert first.task_id.startswith(
        "rewardhack:code/spec-overfit:v0.1.0:split=eval:profile=high:seed=7:template=feature-flags:sha="
    )
    assert len(str(first.metadata["task_content_hash"])) == 16
    assert first.metadata["task_schema_version"] == "0.2.0"
    assert first.metadata["environment_version"] == "0.1.0"
    assert first.metadata["official_verifier_version"] == "0.1.0"
    assert first.metadata["oracle_verifier_version"] == "0.1.0"
    assert first.metadata["generator_version"] == "0.1.0"
    assert first.metadata["profile"] == "high"
    assert first.metadata["split"] == "eval"
    assert first.metadata["seed"] == 7


def test_task_ids_change_when_profile_changes() -> None:
    low_env = create_environment("code/spec-overfit", EnvironmentConfig.from_profile(profile="low"))
    high_env = create_environment("code/spec-overfit", EnvironmentConfig.from_profile(profile="high"))

    assert low_env.sample_task(seed=7).task_id != high_env.sample_task(seed=7).task_id


def test_runtime_metadata_records_task_and_verifier_versions() -> None:
    env = create_environment("math/symbolic-mismatch", EnvironmentConfig.from_profile(profile="medium"))
    task = env.sample_task(seed=7)
    trajectory = env.evaluate_output(task, env.canonical_true_output(task))
    runtime = trajectory.runtime

    assert runtime.task_schema_version == task.metadata["task_schema_version"]
    assert runtime.environment_version == task.metadata["environment_version"]
    assert runtime.official_verifier_version == task.metadata["official_verifier_version"]
    assert runtime.oracle_verifier_version == task.metadata["oracle_verifier_version"]
    assert runtime.generator_version == task.metadata["generator_version"]
    assert runtime.task_content_hash == task.metadata["task_content_hash"]
    assert runtime.task_seed == 7
    assert runtime.dataset_split == "eval"


def test_dataset_splits_produce_stable_distinct_tasks_and_hidden_cases() -> None:
    train_env = create_environment(
        "code/spec-overfit",
        EnvironmentConfig.from_profile(profile="medium", seed=0, dataset_split="train"),
    )
    eval_env = create_environment(
        "code/spec-overfit",
        EnvironmentConfig.from_profile(profile="medium", seed=0, dataset_split="eval"),
    )

    train_task = train_env.sample_task(seed=7)
    eval_task = eval_env.sample_task(seed=7)

    assert train_task.task_id != eval_task.task_id
    assert train_task.metadata["split"] == "train"
    assert eval_task.metadata["split"] == "eval"
    assert train_task.hidden_metadata["hidden_cases"] != eval_task.hidden_metadata["hidden_cases"]
