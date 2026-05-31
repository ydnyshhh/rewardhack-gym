from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import types

import pytest

from rewardhack_gym.envs.code import runtime as code_runtime
from rewardhack_gym.envs.code.runtime import (
    DockerBackend,
    ExecutionResult,
    LocalTrustedBackend,
    PrimeSandboxBackend,
    SubprocessBackend,
    compile_submission,
    run_function_cases_sync,
)


def run_subprocess_case(
    source: str,
    *,
    timeout_s: float = 1.0,
    stdout_limit_chars: int = 1_000,
) -> ExecutionResult:
    backend = SubprocessBackend(
        stdout_limit_chars=stdout_limit_chars,
        stderr_limit_chars=1_000,
        max_output_object_size=1_000,
    )
    return backend.run_function_cases_sync(
        source,
        "solve",
        [{"label": "case", "args": [], "expected": "ok"}],
        timeout_s,
        128,
    )


def test_subprocess_backend_runs_function_cases() -> None:
    result = run_subprocess_case(
        "def solve():\n"
        "    return 'ok'\n"
    )

    assert result.status == "passed"
    assert result.backend == "subprocess"
    assert result.case_results[0]["passed"] is True


def test_compile_submission_is_ast_only_and_does_not_execute_top_level_code() -> None:
    result = compile_submission(
        "raise RuntimeError('should not execute')\n"
        "def solve():\n"
        "    return 'ok'\n",
        "solve",
    )

    assert result.symbol == "solve"
    assert result.diagnostics["symbol_found"] is True


def test_subprocess_timeout_watchdog_returns_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    released = threading.Event()

    class FakeProcess:
        pid = 12345
        returncode: int | None = None

        def communicate(self, input: str | None = None) -> tuple[str, str]:
            del input
            released.wait(timeout=1.0)
            return "", ""

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9
            released.set()

    fake_process = FakeProcess()

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        del args, kwargs
        return fake_process

    def fake_kill_process_tree(proc: FakeProcess) -> None:
        proc.kill()

    monkeypatch.setattr(code_runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(code_runtime, "_kill_process_tree", fake_kill_process_tree)

    result = code_runtime._run_worker_command(
        ["python", "-c", "pass"],
        {"source": "", "symbol_name": "solve", "cases": []},
        backend="subprocess",
        timeout_s=0.01,
    )

    assert result.status == "timeout"
    assert fake_process.returncode == -9


@pytest.mark.skipif(
    os.name == "nt",
    reason="Live infinite-loop timeout tests are Linux/CI-only; Windows shells can leave CPU-bound worker children behind.",
)
def test_subprocess_backend_times_out_and_worker_recovers() -> None:
    timeout_result = run_subprocess_case(
        "def solve():\n"
        "    while True:\n"
        "        pass\n",
        timeout_s=0.25,
    )
    recovery_result = run_subprocess_case(
        "def solve():\n"
        "    return 'ok'\n",
        timeout_s=1.0,
    )

    assert timeout_result.status == "timeout"
    assert recovery_result.status == "passed"


def test_subprocess_backend_blocks_imports() -> None:
    result = run_subprocess_case(
        "import os\n"
        "def solve():\n"
        "    return 'ok'\n"
    )

    assert result.status == "runtime_error"
    assert "Import statements are blocked" in result.stderr


def test_subprocess_backend_blocks_filesystem_access() -> None:
    result = run_subprocess_case(
        "def solve():\n"
        "    return open('/etc/passwd').read()\n"
    )

    assert result.status == "runtime_error"
    assert "open" in result.stderr
    assert result.case_results == []


def test_subprocess_backend_blocks_subprocess_escape() -> None:
    result = run_subprocess_case(
        "def solve():\n"
        "    return __import__('subprocess').run(['echo', 'x'])\n"
    )

    assert result.status == "runtime_error"
    assert "__import__" in result.stderr


def test_subprocess_backend_truncates_stdout() -> None:
    result = run_subprocess_case(
        "print('x' * 10_000_000)\n"
        "def solve():\n"
        "    return 'ok'\n",
        stdout_limit_chars=512,
    )

    assert result.status == "passed"
    assert len(result.stdout) < 700
    assert "truncated" in result.stdout


def test_prime_sandbox_backend_runs_through_prime_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    state: dict[str, object] = {"uploads": {}, "deleted": []}

    class FakeWorkerFiles:
        def __init__(self, payload: dict[str, object]) -> None:
            state["payload"] = payload

        def __enter__(self) -> tuple[str, str]:
            return "worker.py", "payload.json"

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            state["worker_files_closed"] = True

    class FakeCreateSandboxRequest:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeAPIClient:
        pass

    class FakeSandboxClient:
        def __init__(self, api_client: FakeAPIClient) -> None:
            state["api_client"] = api_client

        def create(self, request: FakeCreateSandboxRequest) -> types.SimpleNamespace:
            state["request"] = request
            return types.SimpleNamespace(id="sbx_fake")

        def wait_for_creation(self, sandbox_id: str) -> None:
            state["waited"] = sandbox_id

        def upload_file(self, sandbox_id: str, remote_path: str, local_path: str) -> None:
            del sandbox_id
            uploads = state["uploads"]
            assert isinstance(uploads, dict)
            uploads[remote_path] = local_path

        def execute_command(self, sandbox_id: str, command: str, timeout: float) -> types.SimpleNamespace:
            del sandbox_id
            uploads = state["uploads"]
            assert isinstance(uploads, dict)
            state["command"] = command
            state["timeout"] = timeout
            assert "/tmp/rewardhack_worker.py" in uploads
            assert "/tmp/rewardhack_payload.json" in uploads
            stdout = json.dumps(
                {
                    "status": "passed",
                    "case_results": [{"label": "case", "passed": True, "actual": "ok", "expected": "ok"}],
                    "stdout": "",
                    "stderr": "",
                    "duration_seconds": 0.01,
                }
            )
            return types.SimpleNamespace(stdout=stdout, stderr="", exit_code=0)

        def delete(self, sandbox_id: str) -> None:
            deleted = state["deleted"]
            assert isinstance(deleted, list)
            deleted.append(sandbox_id)

    fake_prime_sandboxes = types.ModuleType("prime_sandboxes")
    fake_prime_sandboxes.APIClient = FakeAPIClient
    fake_prime_sandboxes.CreateSandboxRequest = FakeCreateSandboxRequest
    fake_prime_sandboxes.SandboxClient = FakeSandboxClient
    monkeypatch.setitem(sys.modules, "prime_sandboxes", fake_prime_sandboxes)
    monkeypatch.setattr(code_runtime, "_worker_files", FakeWorkerFiles)

    result = PrimeSandboxBackend(
        image="python:test",
        timeout_minutes=7,
        cpu_cores=2,
    ).run_function_cases_sync(
        "def solve():\n"
        "    return 'ok'\n",
        "solve",
        [{"label": "case", "args": [], "expected": "ok"}],
        1.0,
        2048,
    )

    request = state["request"]
    assert isinstance(request, FakeCreateSandboxRequest)
    assert result.status == "passed"
    assert result.backend == "prime_sandbox"
    assert request.kwargs["docker_image"] == "python:test"
    assert request.kwargs["network_access"] is False
    assert request.kwargs["memory_gb"] == 2
    assert request.kwargs["cpu_cores"] == 2
    assert state["payload"]["memory_mb"] == 2048  # type: ignore[index]
    assert state["command"] == "python -I -B /tmp/rewardhack_worker.py < /tmp/rewardhack_payload.json"
    assert state["timeout"] == 1
    assert state["deleted"] == ["sbx_fake"]
    assert state["worker_files_closed"] is True


def test_public_backend_factories_and_missing_prime_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "prime_sandboxes", None)
    trusted = LocalTrustedBackend()
    trusted_result = trusted.run_function_cases_sync(
        "def solve():\n"
        "    return 'ok'\n",
        "solve",
        [{"label": "case", "args": [], "expected": "ok"}],
    )
    prime_result = asyncio.run(
        PrimeSandboxBackend().run_function_cases(
            "def solve():\n"
            "    return 'ok'\n",
            "solve",
            [{"label": "case", "args": [], "expected": "ok"}],
            1.0,
            128,
        )
    )

    assert trusted_result.status == "passed"
    assert prime_result.status == "sandbox_error"
    assert DockerBackend().backend_name == "docker"
    assert run_function_cases_sync(
        "def solve():\n"
        "    return 'ok'\n",
        "solve",
        [{"label": "case", "args": [], "expected": "ok"}],
    ).status == "passed"
