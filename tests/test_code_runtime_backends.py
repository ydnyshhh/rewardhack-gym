from __future__ import annotations

import asyncio

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


def test_public_backend_factories_and_placeholders() -> None:
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
