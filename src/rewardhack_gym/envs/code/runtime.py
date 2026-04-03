from __future__ import annotations

import ast
import builtins
import collections
import copy
import functools
import inspect
import itertools
import math
import re
from dataclasses import dataclass
from typing import Any

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
