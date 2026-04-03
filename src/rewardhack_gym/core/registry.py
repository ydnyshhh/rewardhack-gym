from __future__ import annotations

from collections.abc import Callable

from rewardhack_gym.core.base import ResearchEnvironment
from rewardhack_gym.core.config import EnvironmentConfig

Factory = Callable[[EnvironmentConfig | None], ResearchEnvironment]

_REGISTRY: dict[str, Factory] = {}


def register_environment(name: str, factory: Factory) -> None:
    if name in _REGISTRY:
        raise ValueError(f"Environment {name!r} is already registered.")
    _REGISTRY[name] = factory


def create_environment(name: str, config: EnvironmentConfig | None = None) -> ResearchEnvironment:
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown environment {name!r}. Available: {sorted(_REGISTRY)}") from exc
    return factory(config)


def list_environments() -> list[str]:
    return sorted(_REGISTRY)
