"""Exact, setup-time dimension expressions with fixed-configuration operands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np

from graphion.errors import DeclarationError, SetupError


class _Parameter:
    """Marker shared by declaration and schema records; has no operators."""

    __slots__ = ()
    kind: str


class _Arithmetic:
    __slots__ = ()

    def __mul__(self, other: Operand) -> Expression:
        return Expression("multiply", _operand(self), _operand(other))

    def __rmul__(self, other: Operand) -> Expression:
        return Expression("multiply", _operand(other), _operand(self))

    def __truediv__(self, other: Operand) -> Expression:
        return Expression("divide", _operand(self), _operand(other))


@dataclass(frozen=True, slots=True)
class SizeToken(_Arithmetic):
    name: str

    def __repr__(self) -> str:
        return self.name


C, N = SizeToken("C"), SizeToken("N")


@dataclass(frozen=True, slots=True)
class _Reference:
    owner: type
    name: str


@dataclass(frozen=True, slots=True)
class Expression(_Arithmetic):
    operator: Literal["multiply", "divide"]
    left: Node
    right: Node


type Operand = int | np.integer | Fraction | _Parameter | _Arithmetic
type Node = Fraction | SizeToken | Expression | _Parameter | _Reference


def _exact(value: object) -> Fraction:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("bool is not a dimension operand")
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, Fraction):
        return Fraction(value.numerator, value.denominator)
    raise ValueError("dimension operands must be integers or Fractions")


def _operand(value: object) -> Node:
    if isinstance(value, (SizeToken, Expression)):
        return value
    if isinstance(value, _Parameter):
        if value.kind != "Param":
            raise DeclarationError("dimension expressions may reference only Params")
        return value
    try:
        return _exact(value)
    except ValueError as exc:
        raise DeclarationError(str(exc)) from exc


def bind(node: Node, parameters: Mapping[int, str], owner: type) -> Node:
    """Bind by identity once, retaining names for per-instance override lookup."""
    if isinstance(node, _Parameter):
        if id(node) not in parameters:
            raise DeclarationError("dimension operand must belong to this class or an ancestor")
        return _Reference(owner, parameters[id(node)])
    if isinstance(node, _Reference):
        if not any(node.owner is base for base in owner.__mro__):
            raise DeclarationError("dimension expression must belong to this class or an ancestor")
        return node
    if isinstance(node, Expression):
        return Expression(node.operator, bind(node.left, parameters, owner), bind(node.right, parameters, owner))
    return node


def _evaluate(node: Node, values: Mapping[str, object], channels: int, n: int) -> Fraction:
    if isinstance(node, Fraction):
        return node
    if isinstance(node, SizeToken):
        if node.name == "C":
            return Fraction(int(channels))
        if node.name == "N":
            return Fraction(int(n))
        raise ValueError(f"unknown size token {node.name!r}")
    if isinstance(node, _Reference):
        try:
            return _exact(values[node.name])
        except ValueError as exc:
            raise ValueError(f"Param {node.name}: {exc}") from exc
    if isinstance(node, Expression):
        left = _evaluate(node.left, values, channels, n)
        right = _evaluate(node.right, values, channels, n)
        return left * right if node.operator == "multiply" else left / right
    raise ValueError("unbound dimension operand")


def _format(node: Node) -> str:
    if isinstance(node, Fraction):
        return str(node) if node.denominator == 1 else f"({node})"
    if isinstance(node, (SizeToken, _Reference)):
        return node.name
    if isinstance(node, Expression):
        operator = "*" if node.operator == "multiply" else "/"
        return f"({_format(node.left)} {operator} {_format(node.right)})"
    return "<unbound Param>"


def resolve(expression: Expression, values: Mapping[str, object], channels: int, n: int, label: str) -> int:
    try:
        result = _evaluate(expression, values, channels, n)
    except ZeroDivisionError as exc:
        raise SetupError(f"{label}: dimension expression {_format(expression)} divides by zero") from exc
    except ValueError as exc:
        raise SetupError(f"{label}: dimension expression {_format(expression)}: {exc}") from exc
    if result.denominator != 1 or result < 0:
        raise SetupError(
            f"{label}: dimension expression {_format(expression)} resolved to {result}; expected a non-negative integer"
        )
    return result.numerator


def describe(node: Node) -> dict[str, object]:
    """Serialize an already-bound tree without reading instance configuration."""
    if isinstance(node, Fraction):
        return {"kind": "fraction", "numerator": node.numerator, "denominator": node.denominator}
    if isinstance(node, SizeToken):
        return {"kind": "symbol", "name": node.name}
    if isinstance(node, _Reference):
        return {"kind": "param", "name": node.name}
    if isinstance(node, Expression):
        return {
            "kind": "expression",
            "operator": node.operator,
            "left": describe(node.left),
            "right": describe(node.right),
        }
    raise DeclarationError("cannot describe an unbound dimension operand")
