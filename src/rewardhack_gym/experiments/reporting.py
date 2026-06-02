from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from rewardhack_gym.experiments.utils import atomic_write_text


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def generate_report(
    *,
    run_dir: str | Path,
    metadata: Mapping[str, Any],
    summary: Mapping[str, Any],
    metrics_by_model: Mapping[str, Mapping[str, Any]],
    metrics_by_profile: Mapping[str, Mapping[str, Any]],
    metrics_by_family: Mapping[str, Mapping[str, Any]],
    best_of_n_metrics: Sequence[Mapping[str, Any]] = (),
    false_pass_examples: Sequence[Mapping[str, Any]] = (),
    output_format: str = "markdown",
) -> str:
    if output_format != "markdown":
        raise ValueError("Only markdown reports are currently supported.")
    rows: list[str] = ["# RewardHack-Gym Experiment Report", ""]
    rows.extend(
        [
            "## Run metadata",
            f"- run id: `{metadata.get('run_id')}`",
            f"- experiment type: `{metadata.get('experiment_type')}`",
            f"- timestamp: `{metadata.get('timestamp')}`",
            f"- git commit: `{metadata.get('git_commit')}`",
            f"- config path: `{metadata.get('config_path')}`",
            f"- families: {', '.join(metadata.get('families', []))}",
            f"- profiles: {', '.join(metadata.get('profiles', []))}",
            f"- models: {', '.join(metadata.get('models', []))}",
            f"- split: `{metadata.get('split')}`",
            f"- num tasks: {metadata.get('num_tasks')}",
            "",
            "## Key metrics",
            _table(
                ["scope", "official", "oracle", "false pass", "hack gap", "true pass"],
                [
                    [
                        "overall",
                        _fmt(summary.get("official_pass_rate", 0.0)),
                        _fmt(summary.get("oracle_pass_rate", 0.0)),
                        _fmt(summary.get("false_pass_rate", 0.0)),
                        _fmt(summary.get("mean_hack_gap", 0.0)),
                        _fmt(summary.get("true_pass_rate", 0.0)),
                    ]
                ],
            ),
            "",
        ]
    )
    for title, bucket in (
        ("By model", metrics_by_model),
        ("By profile", metrics_by_profile),
        ("By family", metrics_by_family),
    ):
        rows.extend(
            [
                f"### {title}",
                _table(
                    ["key", "official", "oracle", "false pass", "hack gap", "true pass"],
                    [
                        [
                            key,
                            _fmt(value.get("official_pass_rate", 0.0)),
                            _fmt(value.get("oracle_pass_rate", 0.0)),
                            _fmt(value.get("false_pass_rate", 0.0)),
                            _fmt(value.get("mean_hack_gap", 0.0)),
                            _fmt(value.get("true_pass_rate", 0.0)),
                        ]
                        for key, value in bucket.items()
                    ],
                ),
                "",
            ]
        )
    if best_of_n_metrics:
        rows.extend(
            [
                "## Best-of-N curves",
                _table(
                    ["policy", "N", "official_pass@N", "oracle_pass@N", "hack@N", "hack amplification"],
                    [
                        [
                            item.get("selection_policy"),
                            item.get("N"),
                            _fmt(item.get("official_pass_at_n", 0.0)),
                            _fmt(item.get("oracle_pass_at_n", 0.0)),
                            _fmt(item.get("hack_at_n", 0.0)),
                            _fmt(item.get("best_of_n_hack_amplification", 0.0)),
                        ]
                        for item in best_of_n_metrics
                    ],
                ),
                "",
                "## Main finding template",
                "Under official-only selection, official pass rate may increase as N increases while oracle pass rate lags, decreases, or saturates. If false-pass rate increases, this suggests verifier over-optimization under the controlled proxy objective.",
                "",
            ]
        )
    rows.extend(["## False-pass examples"])
    if not false_pass_examples:
        rows.append("No false-pass examples were saved for this run.")
    else:
        for index, example in enumerate(false_pass_examples[:10], start=1):
            rows.extend(
                [
                    f"### Example {index}",
                    f"- task id: `{example.get('task_id')}`",
                    f"- model: `{example.get('model_id')}`",
                    f"- official score: {_fmt(example.get('official_score', 0.0))}",
                    f"- oracle score: {_fmt(example.get('oracle_score', 0.0))}",
                    f"- exploit labels: {', '.join(example.get('exploit_labels', []))}",
                    f"- semantic failures: {', '.join(example.get('semantic_failures', []))}",
                    "",
                    "Prompt:",
                    "```text",
                    str(example.get("prompt", ""))[:2000],
                    "```",
                    "Selected completion:",
                    "```text",
                    str(example.get("completion", ""))[:2000],
                    "```",
                ]
            )
    rows.extend(
        [
            "",
            "## Limitations",
            "- Code execution backend and sandbox fidelity can affect runtime diagnostics.",
            "- Model coverage depends on configured clients; dry-run modes are only sanity checks.",
            "- Oracle checks are stronger but still synthetic approximations of intended behavior.",
            "- The benchmark is artificial and should not be overclaimed as a complete alignment evaluation.",
            "- No human audit is included by default.",
            "",
            "RewardHack-Gym turns verifier misspecification into a controllable experimental variable. These runners measure how models behave as optimization pressure increases against the official verifier. If official pass rate improves while oracle pass rate lags or false-pass rate increases, this indicates proxy-objective over-optimization in the controlled benchmark.",
        ]
    )
    report = "\n".join(rows) + "\n"
    atomic_write_text(Path(run_dir) / "report.md", report)
    return report

