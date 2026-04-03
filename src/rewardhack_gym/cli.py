from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rewardhack_gym import create_environment, list_environments
from rewardhack_gym.analysis import summarize_trajectories
from rewardhack_gym.io import read_jsonl, write_jsonl


def read_text_argument(text: str | None, text_file: str | None) -> str:
    if text is not None:
        return text
    if text_file is not None:
        return Path(text_file).read_text(encoding="utf-8")
    raise SystemExit("One of --text or --text-file is required.")


def emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rewardhack-gym", description="RewardHack-Gym CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-envs", help="List registered environments")
    list_parser.set_defaults(handler=cmd_list_envs)

    inspect_parser = subparsers.add_parser("inspect-env", help="Inspect an environment with a sampled task")
    inspect_parser.add_argument("environment")
    inspect_parser.add_argument("--seed", type=int, default=0)
    inspect_parser.add_argument("--include-hidden", action="store_true")
    inspect_parser.set_defaults(handler=cmd_inspect_env)

    sample_parser = subparsers.add_parser("sample-task", help="Sample and print a task")
    sample_parser.add_argument("environment")
    sample_parser.add_argument("--seed", type=int, default=0)
    sample_parser.add_argument("--include-hidden", action="store_true")
    sample_parser.set_defaults(handler=cmd_sample_task)

    eval_parser = subparsers.add_parser("evaluate-output", help="Evaluate a single output against an environment")
    eval_parser.add_argument("environment")
    eval_parser.add_argument("--seed", type=int, default=0)
    eval_parser.add_argument("--text")
    eval_parser.add_argument("--text-file")
    eval_parser.add_argument("--policy-id")
    eval_parser.add_argument("--output")
    eval_parser.add_argument("--include-hidden-task", action="store_true")
    eval_parser.set_defaults(handler=cmd_evaluate_output)

    traces_parser = subparsers.add_parser("evaluate-traces", help="Evaluate a JSONL file of outputs")
    traces_parser.add_argument("environment")
    traces_parser.add_argument("--input", required=True)
    traces_parser.add_argument("--output", required=True)
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
    env = create_environment(args.environment)
    task = env.sample_task(seed=args.seed)
    emit_json(
        {
            "environment": args.environment,
            "task": task.to_dict(include_hidden=args.include_hidden),
            "canonical_true_output": env.canonical_true_output(task),
            "canonical_exploit_output": env.canonical_exploit_output(task),
        }
    )


def cmd_sample_task(args: argparse.Namespace) -> None:
    env = create_environment(args.environment)
    task = env.sample_task(seed=args.seed)
    emit_json(task.to_dict(include_hidden=args.include_hidden))


def cmd_evaluate_output(args: argparse.Namespace) -> None:
    env = create_environment(args.environment)
    task = env.sample_task(seed=args.seed)
    text = read_text_argument(args.text, args.text_file)
    trajectory = env.evaluate_output(task, text, policy_id=args.policy_id)
    payload = trajectory.to_dict(include_hidden_task_metadata=args.include_hidden_task)
    if args.output:
        write_jsonl(args.output, [payload])
    else:
        emit_json(payload)


def cmd_evaluate_traces(args: argparse.Namespace) -> None:
    env = create_environment(args.environment)
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
