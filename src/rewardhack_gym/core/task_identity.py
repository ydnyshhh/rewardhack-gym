from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from typing import Any

from rewardhack_gym.core.models import JSONValue, Task, serialize_value
from rewardhack_gym.core.versions import TASK_ID_STRATEGY


VERSION_METADATA_KEYS = (
    "task_schema_version",
    "environment_version",
    "official_verifier_version",
    "oracle_verifier_version",
    "generator_version",
)


def _metadata_value(config: object, key: str) -> str:
    return str(getattr(config, key))


def task_version_metadata(
    config: object,
    *,
    seed: int,
    profile: str,
) -> dict[str, JSONValue]:
    split = str(getattr(config, "dataset_split", "eval"))
    return {
        "task_schema_version": _metadata_value(config, "task_schema_version"),
        "environment_version": _metadata_value(config, "environment_version"),
        "official_verifier_version": _metadata_value(config, "official_verifier_version"),
        "oracle_verifier_version": _metadata_value(config, "oracle_verifier_version"),
        "generator_version": _metadata_value(config, "generator_version"),
        "profile": profile,
        "split": split,
        "seed": seed,
        "task_id_strategy": TASK_ID_STRATEGY,
    }


def normalized_task_payload(
    task: Task,
    *,
    environment_name: str,
    version_metadata: dict[str, JSONValue],
) -> dict[str, JSONValue]:
    metadata = dict(task.metadata)
    metadata.update(version_metadata)
    return {
        "environment_name": environment_name,
        "family": task.family,
        "prompt": task.prompt,
        "expected_interface": task.expected_interface,
        "difficulty": task.difficulty,
        "exploit_surface": task.exploit_surface.to_dict(),
        "metadata": serialize_value(metadata),
        "hidden_metadata": serialize_value(task.hidden_metadata),
        "tags": serialize_value(task.tags),
    }


def task_content_hash(payload: dict[str, JSONValue]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _template_id(task: Task) -> str:
    for key in ("template", "scenario_id", "exploit_mode"):
        value = task.metadata.get(key)
        if value not in (None, ""):
            return str(value)
    if task.tags:
        return str(task.tags[-1])
    return "task"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or "task"


def stable_task_id(
    *,
    environment_name: str,
    environment_version: str,
    profile: str,
    split: str,
    seed: int,
    template: str,
    digest: str,
) -> str:
    return (
        f"rewardhack:{environment_name}:v{environment_version}:"
        f"split={split}:profile={profile}:seed={seed}:template={_slug(template)}:sha={digest}"
    )


def with_stable_task_identity(
    task: Task,
    *,
    environment_name: str,
    config: Any,
    seed: int,
) -> Task:
    profile = str(getattr(getattr(config, "exploitability"), "level"))
    metadata_config = getattr(config, "metadata", {})
    identity_seed = seed
    if isinstance(metadata_config, dict) and "task_identity_seed" in metadata_config:
        identity_seed = int(metadata_config["task_identity_seed"])
    versions = task_version_metadata(config, seed=identity_seed, profile=profile)
    payload = normalized_task_payload(
        task,
        environment_name=environment_name,
        version_metadata=versions,
    )
    digest = task_content_hash(payload)
    metadata = dict(task.metadata)
    metadata.update(versions)
    metadata["task_content_hash"] = digest
    task_id = stable_task_id(
        environment_name=environment_name,
        environment_version=str(versions["environment_version"]),
        profile=profile,
        split=str(versions["split"]),
        seed=identity_seed,
        template=_template_id(task),
        digest=digest,
    )
    return replace(task, task_id=task_id, metadata=metadata)
