"""Tests for the `calc` reckon-profile tool — harness-xhuc.

Heavy on sandbox-escape negatives: every disallowed AST shape gets a
named test so a future refactor of `_validate_tree` can't silently
re-open an attack surface.
"""

from __future__ import annotations

import pytest

from harness.tools.calc import CalcTool


@pytest.fixture
def tool() -> CalcTool:
    return CalcTool()


# ----------------- spec shape -----------------


def test_spec(tool: CalcTool) -> None:
    spec = tool.spec
    assert spec.name == "calc"
    assert spec.tier == "read"
    assert "expr" in spec.parameters["properties"]


# ----------------- expression eval (happy path) -----------------


def test_arithmetic(tool: CalcTool) -> None:
    assert tool.call(expr="1 + 2") == "1 + 2 -> 3"
    assert tool.call(expr="10 - 4 * 2") == "10 - 4 * 2 -> 2"
    assert "0.5" in tool.call(expr="1 / 2")


def test_power_and_modulo(tool: CalcTool) -> None:
    assert tool.call(expr="2 ** 10") == "2 ** 10 -> 1024"
    assert tool.call(expr="10 % 3") == "10 % 3 -> 1"


def test_floor_div(tool: CalcTool) -> None:
    assert tool.call(expr="10 // 3") == "10 // 3 -> 3"


def test_unary_negative(tool: CalcTool) -> None:
    assert tool.call(expr="-5 + 10") == "-5 + 10 -> 5"


def test_math_functions(tool: CalcTool) -> None:
    assert "7.07" in tool.call(expr="sqrt(50)")
    assert "1024" in tool.call(expr="pow(2, 10)")
    assert tool.call(expr="abs(-7)") == "abs(-7) -> 7"
    assert tool.call(expr="round(3.7)") == "round(3.7) -> 4"
    assert tool.call(expr="floor(2.9)") == "floor(2.9) -> 2"
    assert tool.call(expr="ceil(2.1)") == "ceil(2.1) -> 3"


def test_constants(tool: CalcTool) -> None:
    assert "3.14159" in tool.call(expr="pi")
    assert "2.718" in tool.call(expr="e")


def test_min_max_sum(tool: CalcTool) -> None:
    assert tool.call(expr="min(3, 1, 4)") == "min(3, 1, 4) -> 1"
    assert tool.call(expr="max(3, 1, 4)") == "max(3, 1, 4) -> 4"
    assert tool.call(expr="sum([1, 2, 3])") == "sum([1, 2, 3]) -> 6"


# ----------------- unit conversion -----------------


def test_length_conversion(tool: CalcTool) -> None:
    out = tool.call(expr="5 ft to m")
    assert "1.524" in out
    assert "m (length)" in out


def test_mass_conversion(tool: CalcTool) -> None:
    out = tool.call(expr="100 kg to lb")
    assert "220.46" in out
    assert "(mass)" in out


def test_temperature_celsius_to_fahrenheit(tool: CalcTool) -> None:
    out = tool.call(expr="30 c to f")
    assert "86" in out
    assert "(temperature)" in out


def test_temperature_fahrenheit_to_celsius(tool: CalcTool) -> None:
    out = tool.call(expr="32 f to c")
    assert "0" in out  # 32F == 0C
    assert "(temperature)" in out


def test_temperature_kelvin(tool: CalcTool) -> None:
    out = tool.call(expr="273.15 k to c")
    assert "(temperature)" in out


def test_time_conversion(tool: CalcTool) -> None:
    out = tool.call(expr="2 hours to minutes")
    assert "120" in out
    assert "(time)" in out


def test_volume_conversion(tool: CalcTool) -> None:
    out = tool.call(expr="1 gal to l")
    assert "3.785" in out
    assert "(volume)" in out


def test_conversion_synonyms(tool: CalcTool) -> None:
    # "in" and "->" should both work alongside "to"
    out1 = tool.call(expr="5 ft in m")
    out2 = tool.call(expr="5 ft -> m")
    assert "1.524" in out1
    assert "1.524" in out2


def test_cross_family_rejected(tool: CalcTool) -> None:
    with pytest.raises(ValueError, match="unit families differ"):
        tool.call(expr="5 kg to m")


def test_unknown_unit_rejected(tool: CalcTool) -> None:
    with pytest.raises(ValueError, match="unknown unit"):
        tool.call(expr="5 furlongs to m")


# ----------------- sandbox escape attempts -----------------


def test_rejects_import(tool: CalcTool) -> None:
    with pytest.raises(ValueError, match="disallowed"):
        tool.call(expr="__import__('os')")


def test_rejects_dunder_function(tool: CalcTool) -> None:
    # `__import__` is a name, not in the safe table.
    with pytest.raises(ValueError, match="disallowed"):
        tool.call(expr="__import__")


def test_rejects_attribute_access(tool: CalcTool) -> None:
    with pytest.raises(ValueError, match="disallowed"):
        tool.call(expr="(1).__class__")


def test_rejects_mro_escape(tool: CalcTool) -> None:
    # Classic sandbox-escape: ().__class__.__mro__[1].__subclasses__()
    with pytest.raises(ValueError, match="disallowed"):
        tool.call(expr="().__class__.__mro__[1]")


def test_rejects_subscript_attack(tool: CalcTool) -> None:
    with pytest.raises(ValueError, match="disallowed"):
        tool.call(expr="[1, 2, 3][0]")


def test_rejects_comprehension(tool: CalcTool) -> None:
    with pytest.raises(ValueError, match="disallowed"):
        tool.call(expr="[x*2 for x in [1, 2, 3]]")


def test_rejects_lambda(tool: CalcTool) -> None:
    with pytest.raises(ValueError, match="disallowed"):
        tool.call(expr="(lambda x: x + 1)(5)")


def test_rejects_walrus(tool: CalcTool) -> None:
    with pytest.raises(ValueError, match="disallowed"):
        tool.call(expr="(x := 5) + 3")


def test_rejects_assignment(tool: CalcTool) -> None:
    # Assignment isn't valid in eval-mode parse, but the error should
    # still be a clean ValueError (calc-level), not a raw SyntaxError.
    with pytest.raises(ValueError, match="syntax error"):
        tool.call(expr="x = 5")


def test_rejects_unknown_function(tool: CalcTool) -> None:
    with pytest.raises(ValueError, match="disallowed function"):
        tool.call(expr="open('/etc/passwd')")


def test_rejects_method_call(tool: CalcTool) -> None:
    # `math.sqrt(2)` is a Call on an Attribute — disallowed by the
    # "only top-level names" rule.
    with pytest.raises(ValueError, match="disallowed"):
        tool.call(expr="math.sqrt(2)")


def test_rejects_unknown_name(tool: CalcTool) -> None:
    with pytest.raises(ValueError, match="disallowed name"):
        tool.call(expr="some_random_variable + 1")


# ----------------- error surface -----------------


def test_empty_expr_rejected(tool: CalcTool) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        tool.call(expr="")


def test_syntax_error_surfaces_cleanly(tool: CalcTool) -> None:
    with pytest.raises(ValueError, match="syntax error"):
        tool.call(expr="1 +")


def test_division_by_zero_propagates(tool: CalcTool) -> None:
    with pytest.raises(ZeroDivisionError):
        tool.call(expr="1 / 0")
