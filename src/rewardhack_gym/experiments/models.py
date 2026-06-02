from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from rewardhack_gym.experiments.schemas import SamplingConfig


class ModelClient(Protocol):
    def generate(self, prompt: list[dict[str, str]], sampling: SamplingConfig) -> str:
        ...


class DummyModelClient:
    def __init__(self, output: str = "I don't know.") -> None:
        self.output = output

    def generate(self, prompt: list[dict[str, str]], sampling: SamplingConfig) -> str:
        del prompt, sampling
        return self.output


class CallableModelClient:
    def __init__(self, generate_fn: Callable[[list[dict[str, str]], SamplingConfig], str]) -> None:
        self.generate_fn = generate_fn

    def generate(self, prompt: list[dict[str, str]], sampling: SamplingConfig) -> str:
        return self.generate_fn(prompt, sampling)

