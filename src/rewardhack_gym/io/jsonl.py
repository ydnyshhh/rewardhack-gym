from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from rewardhack_gym.core.models import Trajectory


def coerce_record(record: Trajectory | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, Trajectory):
        return record.to_dict()
    return dict(record)


def write_jsonl(path: str | Path, records: Iterable[Trajectory | Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(coerce_record(record), sort_keys=True))
            handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_parquet(path: str | Path, records: Iterable[Trajectory | Mapping[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Parquet export requires the optional `parquet` dependency group. Run `uv sync --extra parquet`."
        ) from exc

    rows = [coerce_record(record) for record in records]
    table = pa.Table.from_pylist(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target)
