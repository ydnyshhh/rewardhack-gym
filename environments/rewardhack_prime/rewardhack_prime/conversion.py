from __future__ import annotations

from collections.abc import Mapping, Sequence

from rewardhack_gym import Task
from rewardhack_gym.core.models import JSONValue, serialize_value


CANONICAL_METADATA_KEYS = {
    "canonical_true_output",
    "canonical_exploit_output",
}
PRIVATE_METADATA_FRAGMENTS = (
    "hidden",
    "oracle",
)


def _is_private_metadata_key(key: object, *, include_canonical_outputs: bool) -> bool:
    normalized = str(key).lower()
    if not include_canonical_outputs and normalized in CANONICAL_METADATA_KEYS:
        return True
    return any(fragment in normalized for fragment in PRIVATE_METADATA_FRAGMENTS)


def _redact_private_metadata(value: JSONValue, *, include_canonical_outputs: bool) -> JSONValue:
    if isinstance(value, Mapping):
        return {
            str(key): _redact_private_metadata(item, include_canonical_outputs=include_canonical_outputs)
            for key, item in value.items()
            if not _is_private_metadata_key(key, include_canonical_outputs=include_canonical_outputs)
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [
            _redact_private_metadata(item, include_canonical_outputs=include_canonical_outputs)
            for item in value
        ]
    return value


def public_metadata_from_task(
    task: Task,
    *,
    include_canonical_outputs: bool = False,
) -> dict[str, JSONValue]:
    metadata = serialize_value(task.metadata)
    if not isinstance(metadata, Mapping):
        return {}
    public_metadata = _redact_private_metadata(
        metadata,
        include_canonical_outputs=include_canonical_outputs,
    )
    return dict(public_metadata) if isinstance(public_metadata, Mapping) else {}


def rewardhack_task_to_vf_task(
    task: Task,
    *,
    include_canonical_outputs: bool = False,
) -> dict[str, object]:
    return {
        "prompt": [{"role": "user", "content": task.prompt}],
        "answer": "",
        "info": {
            "task_id": task.task_id,
            "family": task.family,
            "difficulty": task.difficulty,
            "expected_interface": task.expected_interface,
            "metadata": public_metadata_from_task(
                task,
                include_canonical_outputs=include_canonical_outputs,
            ),
            "tags": list(task.tags),
            "exploit_surface": task.exploit_surface.to_dict(),
        },
    }
