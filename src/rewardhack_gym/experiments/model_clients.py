from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from rewardhack_gym.experiments.schemas import ModelConfig, SamplingConfig


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


@dataclass(frozen=True)
class OpenAICompatibleClient:
    model: str
    base_url: str
    api_key: str
    timeout_s: float = 120.0
    extra_headers: dict[str, str] | None = None

    def generate(self, prompt: list[dict[str, str]], sampling: SamplingConfig) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": prompt,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
        }
        if sampling.stop:
            payload["stop"] = sampling.stop
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.extra_headers:
            headers.update(self.extra_headers)
        url = self.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = _http_error_body(exc)
            raise RuntimeError(f"Model endpoint returned HTTP {exc.code}: {body[:1000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Model endpoint request failed for {url}: {exc}") from exc
        return completion_text_from_response(data)


def completion_text_from_response(data: dict[str, object]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = first.get("text")
            if isinstance(text, str):
                return text
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text
    raise RuntimeError("Model endpoint response did not contain completion text.")


def _http_error_body(exc: urllib.error.HTTPError) -> str:
    body = exc.read()
    if not body and exc.fp is not None:
        body = exc.fp.read()
    if isinstance(body, str):
        return body
    return body.decode("utf-8", errors="replace")


def _required_metadata(model: ModelConfig, key: str) -> str:
    value = model.metadata.get(key)
    if not value:
        raise ValueError(
            f"Model {model.id!r} with provider {model.provider!r} is missing metadata.{key}. "
            f"Add models[].metadata.{key} in the experiment config."
        )
    return str(value)


def _required_env(model: ModelConfig, env_name: str, *, field_name: str) -> str:
    value = os.environ.get(env_name)
    if not value:
        raise ValueError(
            f"Model {model.id!r} with provider {model.provider!r} requires environment variable {env_name!r} "
            f"from {field_name}. Set {env_name} or update models[].{field_name}."
        )
    return value


def build_model_client(model: ModelConfig) -> ModelClient:
    if model.provider in {"dummy", "static"}:
        return DummyModelClient(str(model.metadata.get("output", "I don't know.")))
    if model.provider == "openai_compatible":
        base_url = _required_metadata(model, "base_url")
        api_key_env = _required_metadata(model, "api_key_env")
        api_key = _required_env(model, api_key_env, field_name="metadata.api_key_env")
        model_name = model.model_path or model.id
        if not model_name:
            raise ValueError(f"Model {model.id!r} requires model_path or id.")
        return OpenAICompatibleClient(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            timeout_s=float(model.metadata.get("timeout_s", 120.0)),
            extra_headers={
                str(key): str(value)
                for key, value in dict(model.metadata.get("extra_headers") or {}).items()
            },
        )
    if model.provider == "prime_inference":
        metadata = dict(model.metadata)
        base_url = str(metadata.get("base_url", "https://api.pinference.ai/api/v1"))
        api_key_env = str(metadata.get("api_key_env", "PRIME_API_KEY"))
        api_key = _required_env(model, api_key_env, field_name="metadata.api_key_env")
        extra_headers = {str(key): str(value) for key, value in dict(metadata.get("extra_headers") or {}).items()}
        team_id_env = metadata.get("team_id_env")
        if team_id_env and os.environ.get(str(team_id_env)):
            extra_headers["X-Prime-Team-ID"] = str(os.environ[str(team_id_env)])
        model_name = model.model_path or model.id
        if not model_name:
            raise ValueError(f"Model {model.id!r} requires model_path or id.")
        return OpenAICompatibleClient(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            timeout_s=float(metadata.get("timeout_s", 120.0)),
            extra_headers=extra_headers,
        )
    raise ValueError(
        f"Model {model.id!r} has unsupported provider {model.provider!r}. "
        "Supported providers: dummy, static, openai_compatible, prime_inference."
    )
