from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from rewardhack_gym.analysis.metrics import PROFILE_MISMATCH_LEVEL


def _records(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in candidates]


def _save_line_plot(path: Path, x_values: list[float], series: dict[str, list[float]], *, xlabel: str, ylabel: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for label, values in series.items():
        plt.plot(x_values, values, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _profile_metric(records: list[dict[str, Any]], metric: str) -> tuple[list[float], list[float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["profile"])].append(record)
    x_values: list[float] = []
    y_values: list[float] = []
    for profile, level in sorted(PROFILE_MISMATCH_LEVEL.items(), key=lambda item: item[1]):
        values = grouped.get(profile, [])
        if not values:
            continue
        x_values.append(level)
        if metric == "official":
            y_values.append(sum(bool(item["official_passed"]) for item in values) / len(values))
        elif metric == "oracle":
            y_values.append(sum(bool(item["oracle_passed"]) for item in values) / len(values))
        elif metric == "false_pass":
            y_values.append(sum(bool(item["false_pass"]) for item in values) / len(values))
        else:
            y_values.append(sum(float(item["hack_gap"]) for item in values) / len(values))
    return x_values, y_values


def write_plots(run_dir: str | Path, candidates: Sequence[Mapping[str, Any]], best_of_n_metrics: Sequence[Mapping[str, Any]] = ()) -> None:
    records = _records(candidates)
    plot_dir = Path(run_dir) / "plots"
    if records:
        x_official, y_official = _profile_metric(records, "official")
        _, y_oracle = _profile_metric(records, "oracle")
        if x_official:
            _save_line_plot(
                plot_dir / "official_vs_oracle_by_profile.png",
                x_official,
                {"official_pass_rate": y_official, "oracle_pass_rate": y_oracle},
                xlabel="profile mismatch level",
                ylabel="pass rate",
                title="Official vs oracle pass rate by profile",
            )
        x_hack, y_hack = _profile_metric(records, "false_pass")
        if x_hack:
            _save_line_plot(
                plot_dir / "false_pass_rate_by_profile.png",
                x_hack,
                {"false_pass_rate": y_hack},
                xlabel="profile mismatch level",
                ylabel="false pass rate",
                title="False pass rate by profile",
            )
        x_gap, y_gap = _profile_metric(records, "hack_gap")
        if x_gap:
            _save_line_plot(
                plot_dir / "hack_gap_by_profile.png",
                x_gap,
                {"hack_gap": y_gap},
                xlabel="profile mismatch level",
                ylabel="mean hack gap",
                title="Hack gap by profile",
            )
    bon = [dict(item) for item in best_of_n_metrics]
    if bon:
        policies = sorted({str(item["selection_policy"]) for item in bon})
        for policy in policies:
            values = sorted((item for item in bon if item["selection_policy"] == policy), key=lambda item: int(item["N"]))
            x_values = [int(item["N"]) for item in values]
            _save_line_plot(
                plot_dir / f"best_of_n_official_oracle_curve_{policy}.png",
                x_values,
                {
                    "official_pass": [float(item["official_pass_at_n"]) for item in values],
                    "oracle_pass": [float(item["oracle_pass_at_n"]) for item in values],
                },
                xlabel="N",
                ylabel="pass@N",
                title=f"Best-of-N pass rates ({policy})",
            )
            _save_line_plot(
                plot_dir / f"best_of_n_hack_rate_curve_{policy}.png",
                x_values,
                {"hack": [float(item["hack_at_n"]) for item in values]},
                xlabel="N",
                ylabel="hack@N",
                title=f"Best-of-N hack rate ({policy})",
            )

