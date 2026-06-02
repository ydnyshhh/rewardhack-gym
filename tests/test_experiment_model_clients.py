from __future__ import annotations

import json
import urllib.error

import pytest

from rewardhack_gym.experiments.model_clients import (
    OpenAICompatibleClient,
    build_model_client,
    completion_text_from_response,
)
from rewardhack_gym.experiments.schemas import ModelConfig, SamplingConfig


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_client_posts_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["headers"] = dict(request.header_items())
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        seen["timeout"] = timeout
        return FakeResponse({"choices": [{"message": {"content": "hello"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        model="model-a",
        base_url="http://localhost:8000/v1",
        api_key="secret",
        timeout_s=12,
        extra_headers={"X-Test": "yes"},
    )

    output = client.generate([{"role": "user", "content": "Hi"}], SamplingConfig(stop=[]))

    assert output == "hello"
    assert seen["url"] == "http://localhost:8000/v1/chat/completions"
    assert seen["payload"]["model"] == "model-a"
    assert seen["payload"]["messages"] == [{"role": "user", "content": "Hi"}]
    assert "stop" not in seen["payload"]
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert seen["headers"]["X-test"] == "yes"
    assert seen["timeout"] == 12


def test_openai_compatible_client_reports_http_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        raise urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)

    error = urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)
    error.fp = type("Body", (), {"read": lambda self: b"bad key"})()

    def fake_urlopen_with_body(request, timeout):
        del request, timeout
        raise error

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen_with_body)
    client = OpenAICompatibleClient(model="m", base_url="http://localhost/v1", api_key="bad")

    with pytest.raises(RuntimeError, match="HTTP 401: bad key"):
        client.generate([{"role": "user", "content": "Hi"}], SamplingConfig())


def test_completion_parser_supports_fallback_shapes() -> None:
    assert completion_text_from_response({"choices": [{"text": "plain"}]}) == "plain"
    assert completion_text_from_response({"output_text": "output"}) == "output"
    with pytest.raises(RuntimeError, match="did not contain completion text"):
        completion_text_from_response({"choices": [{}]})


def test_build_model_client_prime_defaults_and_team_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_API_KEY", "prime-key")
    monkeypatch.setenv("PRIME_TEAM_ID", "team-a")

    client = build_model_client(
        ModelConfig(
            id="Qwen/Qwen3.5-0.8B",
            provider="prime_inference",
            metadata={"team_id_env": "PRIME_TEAM_ID"},
        )
    )

    assert isinstance(client, OpenAICompatibleClient)
    assert client.base_url == "https://api.pinference.ai/api/v1"
    assert client.api_key == "prime-key"
    assert client.model == "Qwen/Qwen3.5-0.8B"
    assert client.extra_headers == {"X-Prime-Team-ID": "team-a"}


def test_build_model_client_validates_missing_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)

    with pytest.raises(ValueError, match="MISSING_KEY"):
        build_model_client(
            ModelConfig(
                id="local-model",
                provider="openai_compatible",
                metadata={"base_url": "http://localhost:8000/v1", "api_key_env": "MISSING_KEY"},
            )
        )

