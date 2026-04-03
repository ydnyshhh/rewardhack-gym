from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from rewardhack_gym.analysis.metrics import record_value
from rewardhack_gym.core.models import Trajectory


def stable_id(namespace: str, payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(f"{namespace}:{serialized}".encode("utf-8")).hexdigest()[:16]
    return f"{namespace}:{digest}"


def environment_name(record: Trajectory | Mapping[str, Any]) -> str:
    explicit = record_value(record, ("environment",), None)
    if isinstance(explicit, str) and explicit:
        return explicit
    exported = record_value(record, ("environment_name",), None)
    if isinstance(exported, str) and exported:
        return exported
    task_id = record_value(record, ("task", "task_id"), record_value(record, ("task_id",), ""))
    if isinstance(task_id, str) and ":" in task_id:
        return task_id.split(":", 1)[0]
    return str(task_id or "unknown")


def scenario_id(record: Trajectory | Mapping[str, Any]) -> str:
    annotation = record_value(record, ("annotations", "scenario_id"), None)
    if annotation is not None:
        return str(annotation)
    top_level = record_value(record, ("scenario_id",), None)
    if top_level is not None:
        return str(top_level)
    for path in (
        ("task", "metadata", "scenario_id"),
        ("task", "metadata", "scenario_kind"),
        ("task", "metadata", "template"),
    ):
        value = record_value(record, path, None)
        if value is not None:
            return str(value)
    return "unknown"


def exploit_class(record: Trajectory | Mapping[str, Any]) -> str | None:
    annotation = record_value(record, ("annotations", "canonical_exploit_class"), None)
    if annotation is not None:
        return str(annotation)
    top_level = record_value(record, ("exploit_class",), None)
    if top_level is not None:
        return str(top_level)
    for path in (("task", "metadata", "exploit_mode"),):
        value = record_value(record, path, None)
        if value is not None:
            return str(value)
    labels = record_value(record, ("exploit_labels",), [])
    if isinstance(labels, Sequence) and labels:
        return str(labels[0])
    return None


def semantic_failures(record: Trajectory | Mapping[str, Any]) -> tuple[str, ...]:
    labels = record_value(record, ("annotations", "semantic_failures"), record_value(record, ("semantic_failures",), []))
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        return ()
    return tuple(sorted(dict.fromkeys(str(label) for label in labels)))


def environment_profile(record: Trajectory | Mapping[str, Any]) -> str | None:
    runtime_profile = record_value(record, ("runtime", "environment_profile"), None)
    if runtime_profile is not None:
        return str(runtime_profile)
    exported_profile = record_value(record, ("environment_profile",), None)
    if exported_profile is not None:
        return str(exported_profile)
    top_level_profile = record_value(record, ("profile",), None)
    if top_level_profile is not None:
        return str(top_level_profile)
    return None


def canonical_output_type(record: Trajectory | Mapping[str, Any]) -> str | None:
    existing = record_value(record, ("canonical_output_type",), None)
    if isinstance(existing, str) and existing:
        return existing
    final_output = record_value(record, ("final_output",), None)
    if not isinstance(final_output, str):
        return None
    canonical_true = record_value(record, ("task", "metadata", "canonical_true_output"), None)
    if isinstance(canonical_true, str) and final_output == canonical_true:
        return "canonical_true"
    canonical_exploit = record_value(record, ("task", "metadata", "canonical_exploit_output"), None)
    if isinstance(canonical_exploit, str) and final_output == canonical_exploit:
        return "canonical_exploit"
    return None


def outcome_label(record: Trajectory | Mapping[str, Any]) -> str:
    official_passed = bool(record_value(record, ("official_result", "passed"), record_value(record, ("official_passed",), False)))
    oracle_passed = bool(record_value(record, ("oracle_result", "passed"), record_value(record, ("oracle_passed",), False)))
    if official_passed and oracle_passed:
        return "true-pass"
    if official_passed and not oracle_passed:
        return "false-pass"
    if (not official_passed) and oracle_passed:
        return "oracle-only-pass"
    return "clean-failure"


def trace_id(record: Trajectory | Mapping[str, Any]) -> str:
    existing = record_value(record, ("trace_id",), None)
    if isinstance(existing, str) and existing:
        return existing
    payload = {
        "environment_name": environment_name(record),
        "task_id": record_value(record, ("task", "task_id"), None),
        "prompt": record_value(record, ("prompt",), ""),
        "final_output": record_value(record, ("final_output",), ""),
        "policy_id": record_value(record, ("runtime", "policy_id"), None),
        "official_score": record_value(record, ("official_result", "score"), record_value(record, ("official_score",), 0.0)),
        "oracle_score": record_value(record, ("oracle_result", "score"), record_value(record, ("oracle_score",), 0.0)),
    }
    return stable_id("trace", payload)


def scenario_cohort_id(record: Trajectory | Mapping[str, Any]) -> str:
    payload = {
        "environment_name": environment_name(record),
        "family": record_value(record, ("task", "family"), record_value(record, ("family",), "unknown")),
        "scenario_id": scenario_id(record),
        "environment_profile": environment_profile(record),
    }
    return stable_id("scenario-cohort", payload)


def failure_slice_id(record: Trajectory | Mapping[str, Any]) -> str:
    payload = {
        "environment_name": environment_name(record),
        "family": record_value(record, ("task", "family"), record_value(record, ("family",), "unknown")),
        "scenario_id": scenario_id(record),
        "environment_profile": environment_profile(record),
        "exploit_class": exploit_class(record),
        "semantic_failures": list(semantic_failures(record)),
        "outcome_label": outcome_label(record),
        "canonical_output_type": canonical_output_type(record),
    }
    return stable_id("failure-slice", payload)


@dataclass(frozen=True, slots=True)
class MechInterpRecord:
    trace_id: str
    environment_name: str
    family: str
    task_id: str | None
    scenario_id: str
    environment_profile: str | None
    exploit_class: str | None
    semantic_failures: tuple[str, ...]
    official_passed: bool
    official_score: float
    oracle_passed: bool
    oracle_score: float
    outcome_label: str
    prompt: str
    final_output: str
    canonical_output_type: str | None
    scenario_cohort_id: str
    failure_slice_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "environment_name": self.environment_name,
            "family": self.family,
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "environment_profile": self.environment_profile,
            "exploit_class": self.exploit_class,
            "semantic_failures": list(self.semantic_failures),
            "official_passed": self.official_passed,
            "official_score": self.official_score,
            "oracle_passed": self.oracle_passed,
            "oracle_score": self.oracle_score,
            "outcome_label": self.outcome_label,
            "prompt": self.prompt,
            "final_output": self.final_output,
            "canonical_output_type": self.canonical_output_type,
            "scenario_cohort_id": self.scenario_cohort_id,
            "failure_slice_id": self.failure_slice_id,
        }


def build_mech_interp_record(record: Trajectory | Mapping[str, Any]) -> MechInterpRecord:
    family_name = record_value(record, ("task", "family"), record_value(record, ("family",), "unknown"))
    official_passed = bool(record_value(record, ("official_result", "passed"), record_value(record, ("official_passed",), False)))
    oracle_passed = bool(record_value(record, ("oracle_result", "passed"), record_value(record, ("oracle_passed",), False)))
    official_score = float(record_value(record, ("official_result", "score"), record_value(record, ("official_score",), 0.0)))
    oracle_score = float(record_value(record, ("oracle_result", "score"), record_value(record, ("oracle_score",), 0.0)))
    return MechInterpRecord(
        trace_id=trace_id(record),
        environment_name=environment_name(record),
        family=str(family_name),
        task_id=record_value(record, ("task", "task_id"), record_value(record, ("task_id",), None)),
        scenario_id=scenario_id(record),
        environment_profile=environment_profile(record),
        exploit_class=exploit_class(record),
        semantic_failures=semantic_failures(record),
        official_passed=official_passed,
        official_score=official_score,
        oracle_passed=oracle_passed,
        oracle_score=oracle_score,
        outcome_label=outcome_label(record),
        prompt=str(record_value(record, ("prompt",), record_value(record, ("task", "prompt"), ""))),
        final_output=str(record_value(record, ("final_output",), "")),
        canonical_output_type=canonical_output_type(record),
        scenario_cohort_id=scenario_cohort_id(record),
        failure_slice_id=failure_slice_id(record),
    )


def build_mech_interp_records(records: Sequence[Trajectory | Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [build_mech_interp_record(record).to_dict() for record in records]


@dataclass(frozen=True, slots=True)
class MatchedPair:
    pair_id: str
    pair_group_id: str
    match_level: str
    environment_name: str
    family: str
    scenario_id: str
    environment_profile: str | None
    true_pass: MechInterpRecord
    false_pass: MechInterpRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "pair_group_id": self.pair_group_id,
            "match_level": self.match_level,
            "environment_name": self.environment_name,
            "family": self.family,
            "scenario_id": self.scenario_id,
            "environment_profile": self.environment_profile,
            "true_pass": self.true_pass.to_dict(),
            "false_pass": self.false_pass.to_dict(),
        }


def pair_group_id(match_level: str, row: MechInterpRecord) -> str:
    if match_level == "exact-task" and row.task_id is not None:
        payload = {
            "environment_name": row.environment_name,
            "task_id": row.task_id,
            "environment_profile": row.environment_profile,
        }
        return stable_id("pair-group", payload)
    payload = {
        "environment_name": row.environment_name,
        "scenario_id": row.scenario_id,
        "environment_profile": row.environment_profile,
    }
    return stable_id("pair-group", payload)


def pair_id(true_pass: MechInterpRecord, false_pass: MechInterpRecord, match_level: str) -> str:
    payload = {
        "match_level": match_level,
        "true_trace_id": true_pass.trace_id,
        "false_trace_id": false_pass.trace_id,
    }
    return stable_id("pair", payload)


def matched_pair_groups(rows: Sequence[MechInterpRecord], *, match_level: str) -> list[MatchedPair]:
    groups: dict[tuple[str, ...], list[MechInterpRecord]] = {}
    for row in rows:
        if row.outcome_label not in {"true-pass", "false-pass"}:
            continue
        if match_level == "exact-task":
            if row.task_id is None:
                continue
            key = (row.environment_name, row.environment_profile or "", row.task_id)
        else:
            key = (row.environment_name, row.environment_profile or "", row.scenario_id)
        groups.setdefault(key, []).append(row)

    pairs: list[MatchedPair] = []
    for group_rows in groups.values():
        true_rows = sorted((row for row in group_rows if row.outcome_label == "true-pass"), key=lambda row: row.trace_id)
        false_rows = sorted((row for row in group_rows if row.outcome_label == "false-pass"), key=lambda row: row.trace_id)
        for true_row, false_row in zip(true_rows, false_rows, strict=False):
            group_id = pair_group_id(match_level, true_row)
            pairs.append(
                MatchedPair(
                    pair_id=pair_id(true_row, false_row, match_level),
                    pair_group_id=group_id,
                    match_level=match_level,
                    environment_name=true_row.environment_name,
                    family=true_row.family,
                    scenario_id=true_row.scenario_id,
                    environment_profile=true_row.environment_profile,
                    true_pass=true_row,
                    false_pass=false_row,
                )
            )
    return pairs


def build_matched_pairs(
    records: Sequence[Trajectory | Mapping[str, Any]],
    *,
    max_pairs_per_group: int | None = None,
) -> list[dict[str, Any]]:
    rows = [build_mech_interp_record(record) for record in records]
    exact_pairs = matched_pair_groups(rows, match_level="exact-task")
    used_trace_ids = {pair.true_pass.trace_id for pair in exact_pairs} | {pair.false_pass.trace_id for pair in exact_pairs}
    remaining_rows = [row for row in rows if row.trace_id not in used_trace_ids]
    scenario_pairs = matched_pair_groups(remaining_rows, match_level="scenario")
    combined = exact_pairs + scenario_pairs

    if max_pairs_per_group is not None and max_pairs_per_group >= 0:
        limited: list[MatchedPair] = []
        counts: dict[str, int] = {}
        for pair in combined:
            seen = counts.get(pair.pair_group_id, 0)
            if seen >= max_pairs_per_group:
                continue
            counts[pair.pair_group_id] = seen + 1
            limited.append(pair)
        combined = limited

    return [pair.to_dict() for pair in combined]
