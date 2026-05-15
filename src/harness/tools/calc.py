"""calc — deterministic expression evaluator + unit conversions.

The reckon profile's primary math tool. Two input shapes:

    1. Expression:  "sqrt(50) * 12"           -> 84.852813...
    2. Conversion:  "5 ft to m"               -> 1.524 m
                    "100 kg to lb"            -> 220.46 lb
                    "30 c to f"               -> 86.0 f

Safety: expression evaluation walks the AST under a node allowlist
(no Import, no Attribute, no comprehension, no lambda). Function /
name lookup goes through a hand-curated table — no builtin reach-through.

Returns a plain string ('<call> -> <value> <unit>') so the model's
constitution gets a clean, copy-pasteable provenance line.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from harness.tools.base import ToolSpec

# Whitelisted callables. Any Call node whose func resolves outside this
# table is rejected. Built-in identifiers (open, eval, __import__,
# getattr) are NOT in the table and therefore not callable.
_SAFE_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "divmod": divmod,
    "pow": pow,
}

# Whitelisted constants. Any Name node not callable and not in this
# table is rejected.
_SAFE_NAMES: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
}

# AST nodes the walker accepts. Anything outside this set raises.
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.Call,
    ast.Name,
    # Operators (passed through as type tags on BinOp / UnaryOp / etc.).
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Load,
)


def _validate_tree(tree: ast.AST) -> None:
    """Walk the parse tree; raise ValueError on any non-whitelisted
    construct. This is the core sandbox boundary."""
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"disallowed AST node: {type(node).__name__} — calc "
                f"accepts arithmetic + whitelisted functions only "
                f"(see _SAFE_FUNCS in calc.py for the full list)"
            )
        if isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, ast.Name):
                raise ValueError(
                    "disallowed call form: only top-level names allowed "
                    "(e.g. `sqrt(2)`, not `m.sqrt(2)`)"
                )
            if func.id not in _SAFE_FUNCS:
                raise ValueError(
                    f"disallowed function {func.id!r} — pick from {sorted(_SAFE_FUNCS)!r}"
                )
        if isinstance(node, ast.Name) and node.id not in _SAFE_NAMES and node.id not in _SAFE_FUNCS:
            raise ValueError(
                f"disallowed name {node.id!r} — calc has no "
                f"variables; only constants {sorted(_SAFE_NAMES)!r} "
                f"and functions {sorted(_SAFE_FUNCS)!r}"
            )


def _eval_safe(expr: str) -> Any:
    """Parse + validate + compile + eval. Raises ValueError on any
    sandbox violation, SyntaxError on a malformed expression."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"calc: syntax error in {expr!r}: {exc.msg}") from exc
    _validate_tree(tree)
    # An empty globals dict + the safe names/funcs as locals keeps the
    # eval namespace closed. `__builtins__` is set to {} explicitly so
    # the bytecode interpreter doesn't reach for the host builtins.
    namespace: dict[str, Any] = {**_SAFE_FUNCS, **_SAFE_NAMES}
    code = compile(tree, "<calc>", "eval")
    return eval(code, {"__builtins__": {}}, namespace)  # noqa: S307


# Unit tables. Length / mass / time / volume are multiplicative
# (base * factor); temperature is affine and handled separately.
_LENGTH_TO_M: dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "yd": 0.9144,
    "yard": 0.9144,
    "yards": 0.9144,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
}

_MASS_TO_KG: dict[str, float] = {
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "g": 0.001,
    "gram": 0.001,
    "grams": 0.001,
    "mg": 1e-6,
    "milligram": 1e-6,
    "milligrams": 1e-6,
    "lb": 0.45359237,
    "lbs": 0.45359237,
    "pound": 0.45359237,
    "pounds": 0.45359237,
    "oz": 0.028349523125,
    "ounce": 0.028349523125,
    "ounces": 0.028349523125,
}

_TIME_TO_S: dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "day": 86400.0,
    "days": 86400.0,
    "week": 604800.0,
    "weeks": 604800.0,
}

_VOLUME_TO_L: dict[str, float] = {
    "l": 1.0,
    "liter": 1.0,
    "liters": 1.0,
    "litre": 1.0,
    "litres": 1.0,
    "ml": 0.001,
    "milliliter": 0.001,
    "milliliters": 0.001,
    "gal": 3.785411784,
    "gallon": 3.785411784,
    "gallons": 3.785411784,
    "qt": 0.946352946,
    "quart": 0.946352946,
    "quarts": 0.946352946,
    "pt": 0.473176473,
    "pint": 0.473176473,
    "pints": 0.473176473,
    "cup": 0.2365882365,
    "cups": 0.2365882365,
}

_TEMP_UNITS: frozenset[str] = frozenset({"c", "f", "k", "celsius", "fahrenheit", "kelvin"})

_UNIT_FAMILIES: tuple[tuple[str, dict[str, float]], ...] = (
    ("length", _LENGTH_TO_M),
    ("mass", _MASS_TO_KG),
    ("time", _TIME_TO_S),
    ("volume", _VOLUME_TO_L),
)

_CONVERT_PATTERN = re.compile(
    r"^\s*"
    r"(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"\s+(?P<src>[a-zA-Z]+)"
    r"\s+(?:to|in|->)\s+"
    r"(?P<dst>[a-zA-Z]+)"
    r"\s*$"
)


def _find_family(unit: str) -> tuple[str, dict[str, float]] | None:
    unit_lc = unit.lower()
    for name, table in _UNIT_FAMILIES:
        if unit_lc in table:
            return name, table
    if unit_lc in _TEMP_UNITS:
        return "temperature", {}
    return None


def _convert_temp(value: float, src: str, dst: str) -> float:
    """Affine temperature conversion. Internal base is celsius."""
    src_lc, dst_lc = src.lower(), dst.lower()
    aliases = {"celsius": "c", "fahrenheit": "f", "kelvin": "k"}
    src_lc = aliases.get(src_lc, src_lc)
    dst_lc = aliases.get(dst_lc, dst_lc)
    if src_lc == "c":
        celsius = value
    elif src_lc == "f":
        celsius = (value - 32.0) * 5.0 / 9.0
    elif src_lc == "k":
        celsius = value - 273.15
    else:
        raise ValueError(f"unknown temperature unit {src!r}")
    if dst_lc == "c":
        return celsius
    if dst_lc == "f":
        return celsius * 9.0 / 5.0 + 32.0
    if dst_lc == "k":
        return celsius + 273.15
    raise ValueError(f"unknown temperature unit {dst!r}")


def _maybe_convert(expr: str) -> str | None:
    """Detect and execute a unit conversion. Returns formatted result
    or None when the expression is not a conversion."""
    m = _CONVERT_PATTERN.match(expr)
    if not m:
        return None
    value = float(m.group("value"))
    src = m.group("src")
    dst = m.group("dst")

    src_family = _find_family(src)
    dst_family = _find_family(dst)
    if src_family is None:
        raise ValueError(f"unknown unit {src!r} — see calc unit tables")
    if dst_family is None:
        raise ValueError(f"unknown unit {dst!r} — see calc unit tables")
    if src_family[0] != dst_family[0]:
        raise ValueError(
            f"unit families differ: {src!r} is {src_family[0]}, "
            f"{dst!r} is {dst_family[0]} — cannot convert"
        )

    family_name = src_family[0]
    if family_name == "temperature":
        result = _convert_temp(value, src, dst)
    else:
        table = src_family[1]
        result = value * table[src.lower()] / table[dst.lower()]
    return f"{value} {src} -> {result:g} {dst} ({family_name})"


@dataclass
class CalcTool:
    """Deterministic expression evaluator + unit conversion.

    Two input shapes (see module docstring):
      - "<number> <unit> to <unit>"  — unit conversion
      - "<arithmetic expression>"    — evaluated against the safe-AST
                                         walker
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calc",
            description=(
                "Evaluate an arithmetic expression OR convert units. "
                "Expression form: 'sqrt(50) * 12', '2 ** 10 + 24'. "
                "Conversion form: '5 ft to m', '100 kg to lb', "
                "'30 c to f'. Functions: "
                f"{', '.join(sorted(_SAFE_FUNCS))}. "
                f"Constants: {', '.join(sorted(_SAFE_NAMES))}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expr": {
                        "type": "string",
                        "description": ("Expression or '<number> <unit> to <unit>' conversion."),
                    },
                },
                "required": ["expr"],
            },
            tier="read",
            display_name="Calculator",
        )

    def call(self, *, expr: str) -> str:
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError("calc: expr must be a non-empty string")
        text = expr.strip()

        converted = _maybe_convert(text)
        if converted is not None:
            return converted

        result = _eval_safe(text)
        # Format: floats with %g (avoids trailing zeros, no exponent
        # for everyday magnitudes). Ints stay ints. Other types fall
        # through to repr — useful for tuples / lists from divmod / etc.
        if isinstance(result, bool):
            # bool is a subclass of int; report explicitly to avoid
            # the user reading "1" / "0" as a unitless number.
            return f"{text} -> {result}"
        if isinstance(result, int):
            return f"{text} -> {result}"
        if isinstance(result, float):
            if math.isnan(result):
                return f"{text} -> nan"
            return f"{text} -> {result:g}"
        return f"{text} -> {result!r}"


__all__ = ["CalcTool"]
