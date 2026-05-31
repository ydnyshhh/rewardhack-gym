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
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
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
DEFAULT_PRIME_SANDBOX_IMAGE = "python:3.12-slim"
DEFAULT_PRIME_SANDBOX_TIMEOUT_MINUTES = 10
DEFAULT_PRIME_SANDBOX_CPU_CORES = 1

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

    available_symbols = [
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    if symbol_name not in available_symbols:
        return CompilationResult(
            symbol=None,
            diagnostics={
                "syntax_ok": True,
                "symbol_found": False,
                "available_symbols": sorted(available_symbols),
                "trust_model": TRUST_MODEL,
            },
        )
    return CompilationResult(
        symbol=symbol_name,
        diagnostics={
            "syntax_ok": True,
            "symbol_found": True,
            "trust_model": TRUST_MODEL,
        },
    )


def _compile_submission_trusted(source: str, symbol_name: str) -> CompilationResult:
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


def _project_records(
    records: list[dict[str, Any]],
    keys: list[str],
    *,
    order_key: str,
    preserve_order: bool,
) -> list[dict[str, Any]]:
    projected = [{key: record.get(key) for key in keys} for record in records]
    if preserve_order:
        return projected
    return sorted(projected, key=lambda item: str(item[order_key]))


def _run_history_scenario(cls: type[Any], scenario: dict[str, Any]) -> dict[str, Any]:
    instance = cls(int(scenario["capacity"]))
    observations: list[dict[str, Any]] = []
    for step in scenario["ops"]:
        method = getattr(instance, step["method"])
        result = method(*step.get("args", []))
        if "expect" in step:
            observations.append(
                {
                    "method": step["method"],
                    "actual": result,
                    "expected": step["expect"],
                    "passed": result == step["expect"],
                }
            )
    return {
        "passed": all(item["passed"] for item in observations),
        "observations": observations,
    }


def _run_case_in_process(target: Any, case: dict[str, Any]) -> dict[str, Any]:
    mode = case.get("mode", "function")
    try:
        if mode == "function":
            return call_function_case(target, case)
        if mode == "non_mutation":
            args = copy.deepcopy(case.get("args", []))
            kwargs = copy.deepcopy(case.get("kwargs", {}))
            before = copy.deepcopy(args[0]) if args else None
            target(*args, **kwargs)
            after = args[0] if args else None
            return {
                "label": case["label"],
                "passed": after == before,
                "input_before": _json_safe(before),
                "input_after": _json_safe(after),
                "mutated": after != before,
            }
        if mode == "schema_projected":
            actual = target(*copy.deepcopy(case.get("args", [])), **copy.deepcopy(case.get("kwargs", {})))
            actual_projected = _project_records(
                actual,
                list(case["projection_keys"]),
                order_key=str(case["order_key"]),
                preserve_order=bool(case["preserve_order"]),
            )
            expected_projected = _project_records(
                copy.deepcopy(case["expected"]),
                list(case["projection_keys"]),
                order_key=str(case["order_key"]),
                preserve_order=bool(case["preserve_order"]),
            )
            return {
                "label": case["label"],
                "passed": actual_projected == expected_projected,
                "actual": _json_safe(actual_projected),
                "expected": _json_safe(expected_projected),
            }
        if mode == "schema_aliasing":
            schema = case["schema"]
            input_records = copy.deepcopy(case["input"])
            before = copy.deepcopy(input_records)
            output_records = target(input_records)
            if output_records:
                output_records[0][schema["nested_key"]]["mode"] = "mutated"
                output_records[0][schema["collection_key"]].append("mutated")
            return {
                "label": case["label"],
                "passed": input_records == before,
                "input_before": _json_safe(before),
                "input_after": _json_safe(input_records),
            }
        if mode == "history_scenario":
            return {"label": case["label"], **_run_history_scenario(target, case)}
        if mode == "history_aliasing":
            instance = target(3)
            instance.push("a")
            instance.push("b")
            snapshot = instance.snapshot()
            snapshot.append("mutated")
            after_snapshot = instance.snapshot()
            expected = ["a", "b"]
            return {
                "label": case["label"],
                "passed": after_snapshot == expected,
                "snapshot_after_external_mutation": _json_safe(after_snapshot),
                "expected": expected,
            }
        raise ValueError(f"Unknown execution case mode {mode!r}.")
    except MemoryError:
        raise
    except Exception as exc:
        return {
            "label": case.get("label", "<unlabeled>"),
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "expected": _json_safe(case.get("expected")),
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
        result = _compile_submission_trusted(source, symbol_name)
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
        case_results = [_run_case_in_process(result.symbol, case) for case in cases]
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


def project_records(records, keys, *, order_key, preserve_order):
    projected = [{key: record.get(key) for key in keys} for record in records]
    if preserve_order:
        return projected
    return sorted(projected, key=lambda item: str(item[order_key]))


def run_history_scenario(cls, scenario, max_object_chars):
    instance = cls(int(scenario["capacity"]))
    observations = []
    for step in scenario["ops"]:
        method = getattr(instance, step["method"])
        result = method(*step.get("args", []))
        if "expect" in step:
            observations.append(
                {
                    "method": step["method"],
                    "actual": json_safe(result, max_object_chars),
                    "expected": json_safe(step["expect"], max_object_chars),
                    "passed": result == step["expect"],
                }
            )
    return {"passed": all(item["passed"] for item in observations), "observations": observations}


def run_case(target, case, max_object_chars):
    mode = case.get("mode", "function")
    try:
        if mode == "function":
            return call_case(target, case, max_object_chars)
        if mode == "non_mutation":
            args = copy.deepcopy(case.get("args", []))
            kwargs = copy.deepcopy(case.get("kwargs", {}))
            before = copy.deepcopy(args[0]) if args else None
            target(*args, **kwargs)
            after = args[0] if args else None
            return {
                "label": case["label"],
                "passed": after == before,
                "input_before": json_safe(before, max_object_chars),
                "input_after": json_safe(after, max_object_chars),
                "mutated": after != before,
            }
        if mode == "schema_projected":
            actual = target(*copy.deepcopy(case.get("args", [])), **copy.deepcopy(case.get("kwargs", {})))
            actual_projected = project_records(
                actual,
                list(case["projection_keys"]),
                order_key=str(case["order_key"]),
                preserve_order=bool(case["preserve_order"]),
            )
            expected_projected = project_records(
                copy.deepcopy(case["expected"]),
                list(case["projection_keys"]),
                order_key=str(case["order_key"]),
                preserve_order=bool(case["preserve_order"]),
            )
            return {
                "label": case["label"],
                "passed": actual_projected == expected_projected,
                "actual": json_safe(actual_projected, max_object_chars),
                "expected": json_safe(expected_projected, max_object_chars),
            }
        if mode == "schema_aliasing":
            schema = case["schema"]
            input_records = copy.deepcopy(case["input"])
            before = copy.deepcopy(input_records)
            output_records = target(input_records)
            if output_records:
                output_records[0][schema["nested_key"]]["mode"] = "mutated"
                output_records[0][schema["collection_key"]].append("mutated")
            return {
                "label": case["label"],
                "passed": input_records == before,
                "input_before": json_safe(before, max_object_chars),
                "input_after": json_safe(input_records, max_object_chars),
            }
        if mode == "history_scenario":
            return {"label": case["label"], **run_history_scenario(target, case, max_object_chars)}
        if mode == "history_aliasing":
            instance = target(3)
            instance.push("a")
            instance.push("b")
            snapshot = instance.snapshot()
            snapshot.append("mutated")
            after_snapshot = instance.snapshot()
            expected = ["a", "b"]
            return {
                "label": case["label"],
                "passed": after_snapshot == expected,
                "snapshot_after_external_mutation": json_safe(after_snapshot, max_object_chars),
                "expected": expected,
            }
        raise ValueError(f"Unknown execution case mode {mode!r}.")
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
                run_case(target, case, max_object_chars)
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
            [_worker_python_executable(), "-I", "-B", "-c", _SUBPROCESS_WORKER_SOURCE],
            payload,
            backend=self.backend_name,
            timeout_s=timeout_s,
            stdout_limit_chars=self.stdout_limit_chars,
            stderr_limit_chars=self.stderr_limit_chars,
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
            stdout_limit_chars=self.stdout_limit_chars,
            stderr_limit_chars=self.stderr_limit_chars,
        )


class PrimeSandboxBackend:
    backend_name = "prime_sandbox"

    def __init__(
        self,
        *,
        image: str = DEFAULT_PRIME_SANDBOX_IMAGE,
        timeout_minutes: int = DEFAULT_PRIME_SANDBOX_TIMEOUT_MINUTES,
        cpu_cores: int = DEFAULT_PRIME_SANDBOX_CPU_CORES,
        stdout_limit_chars: int = DEFAULT_STDOUT_LIMIT_CHARS,
        stderr_limit_chars: int = DEFAULT_STDERR_LIMIT_CHARS,
        max_output_object_size: int = DEFAULT_MAX_OUTPUT_OBJECT_SIZE,
    ) -> None:
        self.image = image
        self.timeout_minutes = timeout_minutes
        self.cpu_cores = cpu_cores
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
        try:
            from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest
        except ImportError as exc:
            return self._missing_sdk_result(exc)

        started = time.perf_counter()
        sandbox_id: str | None = None
        with _worker_files(
            _build_worker_payload(
                source,
                symbol_name,
                cases,
                memory_mb=memory_mb,
                stdout_limit_chars=self.stdout_limit_chars,
                stderr_limit_chars=self.stderr_limit_chars,
                max_output_object_size=self.max_output_object_size,
            )
        ) as (worker_path, payload_path):
            try:
                async with AsyncSandboxClient() as client:
                    request = CreateSandboxRequest(**self._create_request_kwargs(memory_mb))
                    sandbox = await client.create(request)
                    sandbox_id = str(sandbox.id)
                    await client.wait_for_creation(sandbox_id)
                    await client.upload_file(sandbox_id, "/tmp/rewardhack_worker.py", str(worker_path))
                    await client.upload_file(sandbox_id, "/tmp/rewardhack_payload.json", str(payload_path))
                    command_result = await client.execute_command(
                        sandbox_id,
                        "python -I -B /tmp/rewardhack_worker.py < /tmp/rewardhack_payload.json",
                        timeout=_prime_command_timeout_seconds(timeout_s),
                    )
            except Exception as exc:
                return _prime_exception_result(exc, started, self.backend_name)
            finally:
                if sandbox_id is not None:
                    try:
                        await client.delete(sandbox_id)  # type: ignore[possibly-undefined]
                    except Exception:
                        pass

        return _execution_result_from_worker_stdout(
            str(getattr(command_result, "stdout", "")),
            str(getattr(command_result, "stderr", "")),
            int(getattr(command_result, "exit_code", getattr(command_result, "return_code", 0))),
            started=started,
            backend=self.backend_name,
            stdout_limit_chars=self.stdout_limit_chars,
            stderr_limit_chars=self.stderr_limit_chars,
        )

    def run_function_cases_sync(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
        memory_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    ) -> ExecutionResult:
        try:
            from prime_sandboxes import APIClient, CreateSandboxRequest, SandboxClient
        except ImportError as exc:
            return self._missing_sdk_result(exc)

        started = time.perf_counter()
        sandbox_id: str | None = None
        with _worker_files(
            _build_worker_payload(
                source,
                symbol_name,
                cases,
                memory_mb=memory_mb,
                stdout_limit_chars=self.stdout_limit_chars,
                stderr_limit_chars=self.stderr_limit_chars,
                max_output_object_size=self.max_output_object_size,
            )
        ) as (worker_path, payload_path):
            try:
                client = SandboxClient(APIClient())
                request = CreateSandboxRequest(**self._create_request_kwargs(memory_mb))
                sandbox = client.create(request)
                sandbox_id = str(sandbox.id)
                client.wait_for_creation(sandbox_id)
                client.upload_file(sandbox_id, "/tmp/rewardhack_worker.py", str(worker_path))
                client.upload_file(sandbox_id, "/tmp/rewardhack_payload.json", str(payload_path))
                command_result = client.execute_command(
                    sandbox_id,
                    "python -I -B /tmp/rewardhack_worker.py < /tmp/rewardhack_payload.json",
                    timeout=_prime_command_timeout_seconds(timeout_s),
                )
            except Exception as exc:
                return _prime_exception_result(exc, started, self.backend_name)
            finally:
                if sandbox_id is not None:
                    try:
                        client.delete(sandbox_id)  # type: ignore[possibly-undefined]
                    except Exception:
                        pass

        return _execution_result_from_worker_stdout(
            str(getattr(command_result, "stdout", "")),
            str(getattr(command_result, "stderr", "")),
            int(getattr(command_result, "exit_code", getattr(command_result, "return_code", 0))),
            started=started,
            backend=self.backend_name,
            stdout_limit_chars=self.stdout_limit_chars,
            stderr_limit_chars=self.stderr_limit_chars,
        )

    def _create_request_kwargs(self, memory_mb: int) -> dict[str, Any]:
        return {
            "name": f"rewardhack-run-{os.getpid()}",
            "docker_image": self.image,
            "labels": ["rewardhack-gym", "code-execution"],
            "timeout_minutes": self.timeout_minutes,
            "network_access": False,
            "cpu_cores": self.cpu_cores,
            "memory_gb": max(1, math.ceil(memory_mb / 1024)),
        }

    def _missing_sdk_result(self, exc: ImportError) -> ExecutionResult:
        return ExecutionResult(
            status="sandbox_error",
            case_results=[],
            stdout="",
            stderr=(
                "PrimeSandboxBackend requires the optional prime-sandboxes SDK. "
                "Install rewardhack-gym with the prime-sandbox extra or install prime-sandboxes directly. "
                f"Original error: {exc}"
            ),
            duration_seconds=0.0,
            backend=self.backend_name,
        )


def _build_worker_payload(
    source: str,
    symbol_name: str,
    cases: list[dict[str, Any]],
    *,
    memory_mb: int,
    stdout_limit_chars: int,
    stderr_limit_chars: int,
    max_output_object_size: int,
) -> dict[str, Any]:
    return {
        "source": source,
        "symbol_name": symbol_name,
        "cases": cases,
        "memory_mb": memory_mb,
        "stdout_limit_chars": stdout_limit_chars,
        "stderr_limit_chars": stderr_limit_chars,
        "max_output_object_size": max_output_object_size,
    }


def _prime_command_timeout_seconds(timeout_s: float) -> int:
    return max(1, math.ceil(timeout_s))


class _worker_files:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> tuple[Path, Path]:
        self._tmp = _temporary_directory(prefix="rewardhack_prime_exec_")
        directory = Path(self._tmp.name)
        worker_path = directory / "worker.py"
        payload_path = directory / "payload.json"
        worker_path.write_text(_SUBPROCESS_WORKER_SOURCE, encoding="utf-8")
        payload_path.write_text(json.dumps(self.payload), encoding="utf-8")
        return worker_path, payload_path

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()


def _execution_temp_dir() -> str | None:
    configured = os.environ.get("REWARDHACK_EXEC_TMPDIR")
    if configured:
        return configured
    if os.name == "nt" and os.path.isdir("C:\\tmp"):
        return "C:\\tmp"
    return None


def _temporary_directory(prefix: str) -> tempfile.TemporaryDirectory[str]:
    preferred_dir = _execution_temp_dir()
    try:
        return tempfile.TemporaryDirectory(
            prefix=prefix,
            dir=preferred_dir,
            ignore_cleanup_errors=True,
        )
    except OSError:
        return tempfile.TemporaryDirectory(prefix=prefix, ignore_cleanup_errors=True)


def _prime_exception_result(exc: Exception, started: float, backend: str) -> ExecutionResult:
    name = type(exc).__name__
    status: ExecutionStatus
    if name in {"CommandTimeoutError", "SandboxTimeoutError"}:
        status = "timeout"
    elif name == "SandboxOOMError":
        status = "memory_limit"
    else:
        status = "sandbox_error"
    return ExecutionResult(
        status=status,
        case_results=[],
        stdout="",
        stderr=f"{name}: {exc}",
        duration_seconds=time.perf_counter() - started,
        backend=backend,
    )


def _execution_result_from_worker_stdout(
    stdout: str,
    stderr: str,
    exit_code: int,
    *,
    started: float,
    backend: str,
    stdout_limit_chars: int,
    stderr_limit_chars: int,
) -> ExecutionResult:
    duration = time.perf_counter() - started
    if exit_code != 0 and not stdout:
        return ExecutionResult(
            status="memory_limit" if exit_code in {-9, 137} else "sandbox_error",
            case_results=[],
            stdout="",
            stderr=_truncate_text(stderr or f"Worker exited with status {exit_code}.", stderr_limit_chars),
            duration_seconds=duration,
            backend=backend,
        )
    try:
        decoded = json.loads(stdout)
    except json.JSONDecodeError:
        return ExecutionResult(
            status="sandbox_error",
            case_results=[],
            stdout=_truncate_text(stdout or "", stdout_limit_chars),
            stderr=_truncate_text(stderr or "Worker produced invalid output.", stderr_limit_chars),
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


def _run_worker_command(
    command: list[str],
    payload: dict[str, Any],
    *,
    backend: str,
    timeout_s: float,
    stdout_limit_chars: int = DEFAULT_STDOUT_LIMIT_CHARS,
    stderr_limit_chars: int = DEFAULT_STDERR_LIMIT_CHARS,
) -> ExecutionResult:
    started = time.perf_counter()
    cwd_manager = _temporary_directory(prefix="rewardhack_exec_")
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
        timed_out = False

        def timeout_worker() -> None:
            nonlocal timed_out
            if proc.poll() is None:
                timed_out = True
                _kill_process_tree(proc)

        timer = threading.Timer(timeout_s, timeout_worker)
        timer.daemon = True
        timer.start()
        try:
            stdout, stderr = proc.communicate(json.dumps(payload))
        finally:
            timer.cancel()

        if timed_out:
            return ExecutionResult(
                status="timeout",
                case_results=[],
                stdout=_truncate_text(stdout or "", stdout_limit_chars),
                stderr=_truncate_text(stderr or "Execution timed out.", stderr_limit_chars),
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
            stderr=_truncate_text(stderr or f"Worker exited with status {proc.returncode}.", stderr_limit_chars),
            duration_seconds=duration,
            backend=backend,
        )
    try:
        decoded = json.loads(stdout)
    except json.JSONDecodeError:
        return ExecutionResult(
            status="sandbox_error",
            case_results=[],
            stdout=_truncate_text(stdout or "", stdout_limit_chars),
            stderr=_truncate_text(stderr or "Worker produced invalid output.", stderr_limit_chars),
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
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5.0,
            )
        except subprocess.TimeoutExpired:
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass


def _worker_python_executable() -> str:
    if os.name == "nt":
        base_executable = getattr(sys, "_base_executable", "")
        if base_executable:
            return str(base_executable)
    return sys.executable


def get_execution_backend(
    name: str | None = None,
    *,
    stdout_limit_chars: int = DEFAULT_STDOUT_LIMIT_CHARS,
    stderr_limit_chars: int = DEFAULT_STDERR_LIMIT_CHARS,
    max_output_object_size: int = DEFAULT_MAX_OUTPUT_OBJECT_SIZE,
    prime_sandbox_image: str = DEFAULT_PRIME_SANDBOX_IMAGE,
    prime_sandbox_timeout_minutes: int = DEFAULT_PRIME_SANDBOX_TIMEOUT_MINUTES,
    prime_sandbox_cpu_cores: int = DEFAULT_PRIME_SANDBOX_CPU_CORES,
) -> ExecutionBackend:
    backend_name = (name or os.environ.get("REWARDHACK_CODE_BACKEND") or "subprocess").lower()
    if backend_name in {"local", "local_trusted", "trusted"}:
        return LocalTrustedBackend()
    if backend_name == "subprocess":
        return SubprocessBackend(
            stdout_limit_chars=stdout_limit_chars,
            stderr_limit_chars=stderr_limit_chars,
            max_output_object_size=max_output_object_size,
        )
    if backend_name == "docker":
        return DockerBackend(
            stdout_limit_chars=stdout_limit_chars,
            stderr_limit_chars=stderr_limit_chars,
            max_output_object_size=max_output_object_size,
        )
    if backend_name in {"prime", "prime_sandbox"}:
        return PrimeSandboxBackend(
            image=prime_sandbox_image,
            timeout_minutes=prime_sandbox_timeout_minutes,
            cpu_cores=prime_sandbox_cpu_cores,
            stdout_limit_chars=stdout_limit_chars,
            stderr_limit_chars=stderr_limit_chars,
            max_output_object_size=max_output_object_size,
        )
    raise ValueError(f"Unknown execution backend {backend_name!r}.")


def execution_settings_from_config(config: Any) -> dict[str, Any]:
    timeout_s = getattr(config, "effective_code_execution_timeout_seconds", None)
    if timeout_s is None:
        timeout_s = getattr(config, "code_execution_timeout_seconds", None)
    if timeout_s is None:
        timeout_s = getattr(config, "max_runtime_seconds", DEFAULT_TIMEOUT_SECONDS)

    backend = get_execution_backend(
        getattr(config, "code_execution_backend", None),
        stdout_limit_chars=int(getattr(config, "code_execution_stdout_limit_chars", DEFAULT_STDOUT_LIMIT_CHARS)),
        stderr_limit_chars=int(getattr(config, "code_execution_stderr_limit_chars", DEFAULT_STDERR_LIMIT_CHARS)),
        max_output_object_size=int(getattr(config, "code_execution_max_output_object_size", DEFAULT_MAX_OUTPUT_OBJECT_SIZE)),
        prime_sandbox_image=str(getattr(config, "prime_sandbox_image", DEFAULT_PRIME_SANDBOX_IMAGE)),
        prime_sandbox_timeout_minutes=int(getattr(config, "prime_sandbox_timeout_minutes", DEFAULT_PRIME_SANDBOX_TIMEOUT_MINUTES)),
        prime_sandbox_cpu_cores=int(getattr(config, "prime_sandbox_cpu_cores", DEFAULT_PRIME_SANDBOX_CPU_CORES)),
    )
    return {
        "timeout_s": float(timeout_s),
        "memory_mb": int(getattr(config, "code_execution_memory_mb", DEFAULT_MEMORY_LIMIT_MB)),
        "backend": backend,
    }


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
