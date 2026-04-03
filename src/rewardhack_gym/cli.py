from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from rewardhack_gym import create_environment, list_environments
from rewardhack_gym.analysis import summarize_trajectories
from rewardhack_gym.core.config import EnvironmentConfig
from rewardhack_gym.io import read_jsonl, write_jsonl


def read_text_argument(text: str | None, text_file: str | None) -> str:
    if text is not None:
        return text
    if text_file is not None:
        return Path(text_file).read_text(encoding="utf-8")
    raise SystemExit("One of --text or --text-file is required.")


def emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def filter_records_by_mode(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "false-pass":
        return [record for record in records if record.get("is_false_pass")]
    if mode == "true-pass":
        return [
            record
            for record in records
            if record.get("official_result", {}).get("passed") and record.get("oracle_result", {}).get("passed")
        ]
    if mode == "official-pass":
        return [record for record in records if record.get("official_result", {}).get("passed")]
    if mode == "oracle-pass":
        return [record for record in records if record.get("oracle_result", {}).get("passed")]
    raise ValueError(f"Unknown cohort mode: {mode}")


def add_environment_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=("low", "medium", "high", "adversarial"), default="medium")
    parser.add_argument("--official-coverage", type=float)
    parser.add_argument("--hidden-adversarial-strength", type=float)
    parser.add_argument("--parser-strictness", type=float)
    parser.add_argument("--process-validation-fraction", type=float)
    parser.add_argument("--domain-awareness", type=float)
    parser.add_argument("--perturbation-robustness", type=float)
    parser.add_argument("--public-example-diversity", type=float)


def build_environment_config(args: argparse.Namespace) -> EnvironmentConfig:
    overrides = {
        "official_coverage_fraction": args.official_coverage,
        "hidden_adversarial_strength": args.hidden_adversarial_strength,
        "parser_strictness": args.parser_strictness,
        "process_validation_fraction": args.process_validation_fraction,
        "domain_awareness": args.domain_awareness,
        "perturbation_robustness": args.perturbation_robustness,
        "public_example_diversity": args.public_example_diversity,
    }
    cleaned_overrides = {key: value for key, value in overrides.items() if value is not None}
    return EnvironmentConfig.from_profile(seed=args.seed, profile=args.profile, exploitability_overrides=cleaned_overrides)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rewardhack-gym", description="RewardHack-Gym CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-envs", help="List registered environments")
    list_parser.set_defaults(handler=cmd_list_envs)

    inspect_parser = subparsers.add_parser("inspect-env", help="Inspect an environment with a sampled task")
    inspect_parser.add_argument("environment")
    inspect_parser.add_argument("--seed", type=int, default=0)
    inspect_parser.add_argument("--include-hidden", action="store_true")
    add_environment_config_arguments(inspect_parser)
    inspect_parser.set_defaults(handler=cmd_inspect_env)

    sample_parser = subparsers.add_parser("sample-task", help="Sample and print a task")
    sample_parser.add_argument("environment")
    sample_parser.add_argument("--seed", type=int, default=0)
    sample_parser.add_argument("--include-hidden", action="store_true")
    add_environment_config_arguments(sample_parser)
    sample_parser.set_defaults(handler=cmd_sample_task)

    sample_batch_parser = subparsers.add_parser("sample-batch", help="Sample many tasks and write them to JSONL")
    sample_batch_parser.add_argument("environment")
    sample_batch_parser.add_argument("--seed", type=int, default=0)
    sample_batch_parser.add_argument("--count", type=int, required=True)
    sample_batch_parser.add_argument("--output", required=True)
    sample_batch_parser.add_argument("--include-hidden", action="store_true")
    sample_batch_parser.add_argument("--include-canonicals", action="store_true")
    add_environment_config_arguments(sample_batch_parser)
    sample_batch_parser.set_defaults(handler=cmd_sample_batch)

    eval_parser = subparsers.add_parser("evaluate-output", help="Evaluate a single output against an environment")
    eval_parser.add_argument("environment")
    eval_parser.add_argument("--seed", type=int, default=0)
    eval_parser.add_argument("--text")
    eval_parser.add_argument("--text-file")
    eval_parser.add_argument("--policy-id")
    eval_parser.add_argument("--output")
    eval_parser.add_argument("--include-hidden-task", action="store_true")
    add_environment_config_arguments(eval_parser)
    eval_parser.set_defaults(handler=cmd_evaluate_output)

    traces_parser = subparsers.add_parser("evaluate-traces", help="Evaluate a JSONL file of outputs")
    traces_parser.add_argument("environment")
    traces_parser.add_argument("--seed", type=int, default=0)
    traces_parser.add_argument("--input", required=True)
    traces_parser.add_argument("--output", required=True)
    traces_parser.add_argument("--summary-output")
    traces_parser.add_argument("--false-pass-output")
    traces_parser.add_argument("--true-pass-output")
    traces_parser.add_argument("--official-pass-output")
    traces_parser.add_argument("--oracle-pass-output")
    add_environment_config_arguments(traces_parser)
    traces_parser.set_defaults(handler=cmd_evaluate_traces)

    stats_parser = subparsers.add_parser("stats", help="Summarize trajectory JSONL files")
    stats_parser.add_argument("--input", required=True)
    stats_parser.set_defaults(handler=cmd_stats)

    export_parser = subparsers.add_parser("export-false-passes", help="Export false-pass traces to a new JSONL")
    export_parser.add_argument("--input", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.set_defaults(handler=cmd_export_false_passes)

    return parser


def cmd_list_envs(args: argparse.Namespace) -> None:
    del args
    emit_json({"environments": list_environments()})


def cmd_inspect_env(args: argparse.Namespace) -> None:
    config = build_environment_config(args)
    env = create_environment(args.environment, config)
    task = env.sample_task(seed=args.seed)
    emit_json(
        {
            "environment": args.environment,
            "profile": config.exploitability.level,
            "exploitability": asdict(config.exploitability),
            "task": task.to_dict(include_hidden=args.include_hidden),
            "canonical_true_output": env.canonical_true_output(task),
            "canonical_exploit_output": env.canonical_exploit_output(task),
        }
    )


def cmd_sample_task(args: argparse.Namespace) -> None:
    env = create_environment(args.environment, build_environment_config(args))
    task = env.sample_task(seed=args.seed)
    emit_json(task.to_dict(include_hidden=args.include_hidden))


def cmd_sample_batch(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise SystemExit("--count must be positive.")
    config = build_environment_config(args)
    env = create_environment(args.environment, config)
    records = []
    for offset in range(args.count):
        seed = args.seed + offset
        task = env.sample_task(seed=seed)
        record: dict[str, Any] = {
            "environment": args.environment,
            "seed": seed,
            "profile": config.exploitability.level,
            "task": task.to_dict(include_hidden=args.include_hidden),
        }
        if args.include_canonicals:
            record["canonical_true_output"] = env.canonical_true_output(task)
            record["canonical_exploit_output"] = env.canonical_exploit_output(task)
        records.append(record)
    write_jsonl(args.output, records)
    emit_json({"written": len(records), "output": args.output, "environment": args.environment})


def cmd_evaluate_output(args: argparse.Namespace) -> None:
    env = create_environment(args.environment, build_environment_config(args))
    task = env.sample_task(seed=args.seed)
    text = read_text_argument(args.text, args.text_file)
    trajectory = env.evaluate_output(task, text, policy_id=args.policy_id)
    payload = trajectory.to_dict(include_hidden_task_metadata=args.include_hidden_task)
    if args.output:
        write_jsonl(args.output, [payload])
    else:
        emit_json(payload)


def cmd_evaluate_traces(args: argparse.Namespace) -> None:
    env = create_environment(args.environment, build_environment_config(args))
    input_records = read_jsonl(args.input)
    output_records = []
    for index, record in enumerate(input_records):
        seed = int(record.get("seed", index))
        task = env.sample_task(seed=seed)
        output = record.get("final_output") or record.get("output") or record.get("text")
        if not isinstance(output, str):
            raise SystemExit(f"Input record {index} is missing a string output field.")
        trajectory = env.evaluate_output(task, output, policy_id=record.get("policy_id"))
        output_records.append(trajectory.to_dict())
    write_jsonl(args.output, output_records)
    if args.summary_output:
        write_json(args.summary_output, summarize_trajectories(output_records).to_dict())
    if args.false_pass_output:
        write_jsonl(args.false_pass_output, filter_records_by_mode(output_records, "false-pass"))
    if args.true_pass_output:
        write_jsonl(args.true_pass_output, filter_records_by_mode(output_records, "true-pass"))
    if args.official_pass_output:
        write_jsonl(args.official_pass_output, filter_records_by_mode(output_records, "official-pass"))
    if args.oracle_pass_output:
        write_jsonl(args.oracle_pass_output, filter_records_by_mode(output_records, "oracle-pass"))


def cmd_stats(args: argparse.Namespace) -> None:
    records = read_jsonl(args.input)
    emit_json(summarize_trajectories(records).to_dict())


def cmd_export_false_passes(args: argparse.Namespace) -> None:
    records = read_jsonl(args.input)
    filtered = [record for record in records if record.get("is_false_pass")]
    write_jsonl(args.output, filtered)
    emit_json({"written": len(filtered), "output": args.output})


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
