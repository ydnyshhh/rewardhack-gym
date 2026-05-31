from __future__ import annotations

import ast
import asyncio
import builtins
import collections
import copy
import functools
import inspect
import itertools
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

ExecutionStatus = Literal[
    "passed",
    "failed",
    "syntax_error",
    "runtime_error",
    "timeout",
    "memory_limit",
    "sandbox_error",
]

DEFAULT_TIMEOUT_SECONDS = 2.0
DEFAULT_MEMORY_LIMIT_MB = 256
DEFAULT_STDOUT_LIMIT_CHARS = 20_000
DEFAULT_STDERR_LIMIT_CHARS = 20_000
DEFAULT_MAX_OUTPUT_OBJECT_SIZE = 20_000

TRUST_MODEL = "trusted-local-only"
TRUST_MODEL_WARNING = (
    "Code execution is for trusted local research use only. "
    "The runtime is intentionally lightweight and is not a security sandbox."
)


SAFE_GLOBALS: dict[str, Any] = {
    "__name__": "__submission__",
    "__builtins__": {
        "__build_class__": builtins.__build_class__,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "object": object,
        "range": range,
        "reversed": reversed,
        "set": set,
        "sorted": sorted,
        "str": str,
        "super": super,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    },
    "collections": collections,
    "copy": copy,
    "functools": functools,
    "itertools": itertools,
    "math": math,
    "re": re,
}


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    status: ExecutionStatus
    case_results: list[dict[str, Any]]
    stdout: str
    stderr: str
    duration_seconds: float
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "case_results": self.case_results,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "backend": self.backend,
        }


class ExecutionBackend(Protocol):
    async def run_function_cases(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float,
        memory_mb: int,
    ) -> ExecutionResult:
        ...


@dataclass(frozen=True, slots=True)
class CompilationResult:
    symbol: Any | None
    diagnostics: dict[str, Any]


def compile_submission(source: str, symbol_name: str) -> CompilationResult:
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        return CompilationResult(
            symbol=None,
            diagnostics={
                "syntax_ok": False,
                "error": f"{exc.msg} at line {exc.lineno}:{exc.offset}",
                "trust_model": TRUST_MODEL,
            },
        )

    namespace: dict[str, Any] = dict(SAFE_GLOBALS)
    try:
        exec(compile(module, "<submission>", "exec"), namespace, namespace)
    except Exception as exc:  # pragma: no cover - exact interpreter messages vary
        return CompilationResult(
            symbol=None,
            diagnostics={
                "syntax_ok": True,
                "execution_ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trust_model": TRUST_MODEL,
            },
        )

    symbol = namespace.get(symbol_name)
    if symbol is None:
        return CompilationResult(
            symbol=None,
            diagnostics={
                "syntax_ok": True,
                "execution_ok": True,
                "symbol_found": False,
                "available_symbols": sorted(name for name in namespace if not name.startswith("__")),
                "trust_model": TRUST_MODEL,
            },
        )
    return CompilationResult(
        symbol=symbol,
        diagnostics={
            "syntax_ok": True,
            "execution_ok": True,
            "symbol_found": True,
            "trust_model": TRUST_MODEL,
        },
    )


def get_ast_signature(source: str, symbol_name: str) -> tuple[str, ...] | None:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == symbol_name:
            return tuple(argument.arg for argument in node.args.args)
        if isinstance(node, ast.ClassDef) and node.name == symbol_name:
            return tuple(
                child.name
                for child in node.body
                if isinstance(child, ast.FunctionDef) and not child.name.startswith("_")
            )
    return None


def call_function_case(fn: Any, case: dict[str, Any]) -> dict[str, Any]:
    args = copy.deepcopy(case.get("args", []))
    kwargs = copy.deepcopy(case.get("kwargs", {}))
    try:
        actual = fn(*args, **kwargs)
        return {
            "label": case["label"],
            "passed": actual == case["expected"],
            "actual": actual,
            "expected": case["expected"],
        }
    except Exception as exc:
        return {
            "label": case["label"],
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "expected": case["expected"],
        }


def describe_callable_signature(symbol: Any) -> str:
    try:
        return str(inspect.signature(symbol))
    except (TypeError, ValueError):
        return "<unavailable>"


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...<truncated {len(value) - limit} chars>"


def _json_safe(value: Any, *, max_chars: int = DEFAULT_MAX_OUTPUT_OBJECT_SIZE) -> Any:
    try:
        encoded = json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return _truncate_text(repr(value), max_chars)
    if len(encoded) <= max_chars:
        return value
    return _truncate_text(repr(value), max_chars)


class LocalTrustedBackend:
    """Current in-process runner. Use only for trusted local research."""

    backend_name = "local_trusted"

    async def run_function_cases(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float,
        memory_mb: int,
    ) -> ExecutionResult:
        del timeout_s, memory_mb
        return self.run_function_cases_sync(source, symbol_name, cases)

    def run_function_cases_sync(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
        memory_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    ) -> ExecutionResult:
        del timeout_s, memory_mb
        started = time.perf_counter()
        result = compile_submission(source, symbol_name)
        if result.symbol is None:
            status: ExecutionStatus = "syntax_error" if not result.diagnostics.get("syntax_ok") else "runtime_error"
            return ExecutionResult(
                status=status,
                case_results=[],
                stdout="",
                stderr=str(result.diagnostics.get("error", result.diagnostics)),
                duration_seconds=time.perf_counter() - started,
                backend=self.backend_name,
            )
        case_results = [call_function_case(result.symbol, case) for case in cases]
        return ExecutionResult(
            status="passed" if all(item.get("passed") for item in case_results) else "failed",
            case_results=case_results,
            stdout="",
            stderr="",
            duration_seconds=time.perf_counter() - started,
            backend=self.backend_name,
        )


_SUBPROCESS_WORKER_SOURCE = r'''
from __future__ import annotations

import ast
import collections
import copy
import functools
import io
import itertools
import json
import math
import re
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

BLOCKED_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "subprocess",
    "vars",
}


def truncate_text(value, limit):
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...<truncated {len(value) - limit} chars>"


def json_safe(value, max_chars):
    try:
        encoded = json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return truncate_text(repr(value), max_chars)
    if len(encoded) <= max_chars:
        return value
    return truncate_text(repr(value), max_chars)


def validate_ast(module):
    for node in ast.walk(module):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "Import statements are blocked by the execution sandbox."
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            return f"Use of {node.id!r} is blocked by the execution sandbox."
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            return f"Dunder attribute access {node.attr!r} is blocked by the execution sandbox."
    return None


def call_case(target, case, max_object_chars):
    args = copy.deepcopy(case.get("args", []))
    kwargs = copy.deepcopy(case.get("kwargs", {}))
    try:
        actual = target(*args, **kwargs)
        expected = case.get("expected")
        return {
            "label": case["label"],
            "passed": actual == expected,
            "actual": json_safe(actual, max_object_chars),
            "expected": json_safe(expected, max_object_chars),
        }
    except MemoryError:
        raise
    except Exception as exc:
        return {
            "label": case.get("label", "<unlabeled>"),
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "expected": json_safe(case.get("expected"), max_object_chars),
        }


def main():
    started = time.perf_counter()
    payload = json.loads(sys.stdin.read())
    stdout_limit = int(payload["stdout_limit_chars"])
    stderr_limit = int(payload["stderr_limit_chars"])
    max_object_chars = int(payload["max_output_object_size"])

    try:
        import resource

        memory_bytes = int(payload["memory_mb"]) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except Exception:
        pass

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    case_results = []
    try:
        module = ast.parse(payload["source"])
    except SyntaxError as exc:
        print(json.dumps({
            "status": "syntax_error",
            "case_results": [],
            "stdout": "",
            "stderr": f"{exc.msg} at line {exc.lineno}:{exc.offset}",
            "duration_seconds": time.perf_counter() - started,
        }))
        return

    validation_error = validate_ast(module)
    if validation_error is not None:
        print(json.dumps({
            "status": "runtime_error",
            "case_results": [],
            "stdout": "",
            "stderr": validation_error,
            "duration_seconds": time.perf_counter() - started,
        }))
        return

    safe_builtins = {
        "__build_class__": __build_class__,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "object": object,
        "print": print,
        "range": range,
        "reversed": reversed,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "super": super,
        "tuple": tuple,
        "type": type,
        "zip": zip,
    }
    namespace = {
        "__name__": "__submission__",
        "__builtins__": safe_builtins,
        "collections": collections,
        "copy": copy,
        "functools": functools,
        "itertools": itertools,
        "math": math,
        "re": re,
    }
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(compile(module, "<submission>", "exec"), namespace, namespace)
            target = namespace[payload["symbol_name"]]
            case_results = [
                call_case(target, case, max_object_chars)
                for case in payload["cases"]
            ]
    except MemoryError:
        status = "memory_limit"
        stderr_buffer.write("MemoryError")
        case_results = []
    except Exception as exc:
        status = "runtime_error"
        stderr_buffer.write(f"{type(exc).__name__}: {exc}")
        case_results = []
    else:
        status = "passed" if all(item.get("passed") for item in case_results) else "failed"

    print(json.dumps({
        "status": status,
        "case_results": case_results,
        "stdout": truncate_text(stdout_buffer.getvalue(), stdout_limit),
        "stderr": truncate_text(stderr_buffer.getvalue(), stderr_limit),
        "duration_seconds": time.perf_counter() - started,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
'''


class SubprocessBackend:
    backend_name = "subprocess"

    def __init__(
        self,
        *,
        stdout_limit_chars: int = DEFAULT_STDOUT_LIMIT_CHARS,
        stderr_limit_chars: int = DEFAULT_STDERR_LIMIT_CHARS,
        max_output_object_size: int = DEFAULT_MAX_OUTPUT_OBJECT_SIZE,
    ) -> None:
        self.stdout_limit_chars = stdout_limit_chars
        self.stderr_limit_chars = stderr_limit_chars
        self.max_output_object_size = max_output_object_size

    async def run_function_cases(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float,
        memory_mb: int,
    ) -> ExecutionResult:
        return await asyncio.to_thread(
            self.run_function_cases_sync,
            source,
            symbol_name,
            cases,
            timeout_s,
            memory_mb,
        )

    def run_function_cases_sync(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
        memory_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    ) -> ExecutionResult:
        payload = {
            "source": source,
            "symbol_name": symbol_name,
            "cases": cases,
            "memory_mb": memory_mb,
            "stdout_limit_chars": self.stdout_limit_chars,
            "stderr_limit_chars": self.stderr_limit_chars,
            "max_output_object_size": self.max_output_object_size,
        }
        return _run_worker_command(
            [sys.executable, "-I", "-B", "-c", _SUBPROCESS_WORKER_SOURCE],
            payload,
            backend=self.backend_name,
            timeout_s=timeout_s,
        )


class DockerBackend(SubprocessBackend):
    backend_name = "docker"

    def __init__(self, *, image: str = "python:3.12-slim", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.image = image

    def run_function_cases_sync(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
        memory_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    ) -> ExecutionResult:
        import shutil

        if shutil.which("docker") is None:
            return ExecutionResult(
                status="sandbox_error",
                case_results=[],
                stdout="",
                stderr="Docker executable is not available.",
                duration_seconds=0.0,
                backend=self.backend_name,
            )
        payload = {
            "source": source,
            "symbol_name": symbol_name,
            "cases": cases,
            "memory_mb": memory_mb,
            "stdout_limit_chars": self.stdout_limit_chars,
            "stderr_limit_chars": self.stderr_limit_chars,
            "max_output_object_size": self.max_output_object_size,
        }
        return _run_worker_command(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--memory",
                f"{memory_mb}m",
                "--cpus",
                "1",
                "-i",
                self.image,
                "python",
                "-I",
                "-B",
                "-c",
                _SUBPROCESS_WORKER_SOURCE,
            ],
            payload,
            backend=self.backend_name,
            timeout_s=timeout_s,
        )


class PrimeSandboxBackend:
    backend_name = "prime_sandbox"

    async def run_function_cases(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float,
        memory_mb: int,
    ) -> ExecutionResult:
        del source, symbol_name, cases, timeout_s, memory_mb
        return ExecutionResult(
            status="sandbox_error",
            case_results=[],
            stdout="",
            stderr="PrimeSandboxBackend is reserved for a future Prime-native sandbox.",
            duration_seconds=0.0,
            backend=self.backend_name,
        )


def _run_worker_command(
    command: list[str],
    payload: dict[str, Any],
    *,
    backend: str,
    timeout_s: float,
) -> ExecutionResult:
    started = time.perf_counter()
    cwd_manager = tempfile.TemporaryDirectory(prefix="rewardhack_exec_", ignore_cleanup_errors=True)
    try:
        cwd = cwd_manager.__enter__()
    except OSError:
        cwd = None
    creationflags = 0
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            creationflags=creationflags,
            **popen_kwargs,
        )
        try:
            stdout, stderr = proc.communicate(json.dumps(payload), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            stdout, stderr = proc.communicate()
            return ExecutionResult(
                status="timeout",
                case_results=[],
                stdout=_truncate_text(stdout or "", DEFAULT_STDOUT_LIMIT_CHARS),
                stderr=_truncate_text(stderr or "Execution timed out.", DEFAULT_STDERR_LIMIT_CHARS),
                duration_seconds=time.perf_counter() - started,
                backend=backend,
            )
    except OSError as exc:
        return ExecutionResult(
            status="sandbox_error",
            case_results=[],
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
            duration_seconds=time.perf_counter() - started,
            backend=backend,
        )
    finally:
        try:
            cwd_manager.cleanup()
        except OSError:
            pass

    duration = time.perf_counter() - started
    if proc.returncode != 0 and not stdout:
        return ExecutionResult(
            status="memory_limit" if proc.returncode in {-9, 137} else "sandbox_error",
            case_results=[],
            stdout="",
            stderr=_truncate_text(stderr or f"Worker exited with status {proc.returncode}.", DEFAULT_STDERR_LIMIT_CHARS),
            duration_seconds=duration,
            backend=backend,
        )
    try:
        decoded = json.loads(stdout)
    except json.JSONDecodeError:
        return ExecutionResult(
            status="sandbox_error",
            case_results=[],
            stdout=_truncate_text(stdout or "", DEFAULT_STDOUT_LIMIT_CHARS),
            stderr=_truncate_text(stderr or "Worker produced invalid output.", DEFAULT_STDERR_LIMIT_CHARS),
            duration_seconds=duration,
            backend=backend,
        )
    worker_stderr = str(decoded.get("stderr", ""))
    if stderr:
        worker_stderr = (worker_stderr + "\n" + stderr).strip()
    return ExecutionResult(
        status=decoded.get("status", "sandbox_error"),
        case_results=list(decoded.get("case_results", [])),
        stdout=str(decoded.get("stdout", "")),
        stderr=worker_stderr,
        duration_seconds=float(decoded.get("duration_seconds", duration)),
        backend=backend,
    )


def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass


def get_execution_backend(name: str | None = None) -> ExecutionBackend:
    backend_name = (name or os.environ.get("REWARDHACK_CODE_BACKEND") or "subprocess").lower()
    if backend_name in {"local", "local_trusted", "trusted"}:
        return LocalTrustedBackend()
    if backend_name == "subprocess":
        return SubprocessBackend()
    if backend_name == "docker":
        return DockerBackend()
    if backend_name in {"prime", "prime_sandbox"}:
        return PrimeSandboxBackend()
    raise ValueError(f"Unknown execution backend {backend_name!r}.")


async def run_function_cases(
    source: str,
    symbol_name: str,
    cases: list[dict[str, Any]],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    backend: ExecutionBackend | None = None,
) -> ExecutionResult:
    selected_backend = backend or get_execution_backend()
    return await selected_backend.run_function_cases(
        source,
        symbol_name,
        cases,
        timeout_s,
        memory_mb,
    )


def run_function_cases_sync(
    source: str,
    symbol_name: str,
    cases: list[dict[str, Any]],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    backend: ExecutionBackend | None = None,
) -> ExecutionResult:
    selected_backend = backend or get_execution_backend()
    if hasattr(selected_backend, "run_function_cases_sync"):
        return selected_backend.run_function_cases_sync(  # type: ignore[attr-defined]
            source,
            symbol_name,
            cases,
            timeout_s,
            memory_mb,
        )
    return asyncio.run(
        selected_backend.run_function_cases(
            source,
            symbol_name,
            cases,
            timeout_s,
            memory_mb,
        )
    )
