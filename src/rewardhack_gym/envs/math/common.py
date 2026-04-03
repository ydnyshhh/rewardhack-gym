from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def require_sympy() -> Any:
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            standard_transformations,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without dependency installation
        raise RuntimeError(
            "RewardHack-Gym math environments require `sympy`. Install dependencies with `uv sync`."
        ) from exc
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    return sp, transformations


def parse_expression(expression: str, variables: tuple[str, ...] = ("x",)) -> Any:
    sp, transformations = require_sympy()
    local_dict = {name: sp.Symbol(name, real=True) for name in variables}
    local_dict.update(
        {
            "pi": sp.pi,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "sqrt": sp.sqrt,
            "Abs": sp.Abs,
        }
    )
    from sympy.parsing.sympy_parser import parse_expr

    return parse_expr(expression.strip(), local_dict=local_dict, transformations=transformations, evaluate=True)


def parse_solution_set(text: str) -> set[Any]:
    cleaned = text.strip()
    if not cleaned:
        return set()
    if cleaned.lower() in {"none", "no solution", "empty"}:
        return set()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        cleaned = cleaned[1:-1]
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    return {parse_expression(part) for part in parts}


@dataclass(frozen=True, slots=True)
class ParsedLine:
    label: str
    value: str | None


def extract_labeled_line(text: str, label: str) -> ParsedLine:
    prefix = f"{label}:"
    for raw_line in text.splitlines():
        if raw_line.strip().startswith(prefix):
            return ParsedLine(label=label, value=raw_line.split(":", 1)[1].strip())
    return ParsedLine(label=label, value=None)
