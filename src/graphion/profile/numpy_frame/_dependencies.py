"""Explicit Input dependencies for Output evaluation."""

import inspect
from collections.abc import Callable
from typing import Any, TypeVar, overload

from graphion.errors import DeclarationError

_Hook = TypeVar("_Hook", bound=Callable[..., Any])


@overload
def needs_inputs(name: None, /) -> Callable[[_Hook], _Hook]: ...


@overload
def needs_inputs(name: str, /, *more: str) -> Callable[[_Hook], _Hook]: ...


def needs_inputs(*names: str | None) -> Callable[[_Hook], _Hook]:
    """Declare Output Input dependencies; pass None alone for no dependencies."""
    if len(names) == 1 and callable(names[0]):
        raise DeclarationError("use @needs_inputs(...) with parentheses; pass None for no Input dependencies")
    if not names:
        raise DeclarationError("needs_inputs requires Input names or None for no dependencies")
    if len(names) == 1 and names[0] is None:
        dependencies: tuple[str, ...] = ()
    else:
        if any(not isinstance(name, str) for name in names):
            raise DeclarationError("needs_inputs accepts Input names as strings, or None alone")
        if len(set(names)) != len(names):
            raise DeclarationError("needs_inputs contains duplicate Input names")
        dependencies = tuple(name for name in names if isinstance(name, str))

    def decorate(hook: _Hook) -> _Hook:
        if not inspect.isfunction(hook) or hook.__name__ != "output":
            raise DeclarationError("needs_inputs applies only to the output instance method")
        if hasattr(hook, "_graphion_input_dependencies"):
            raise DeclarationError("output may have only one needs_inputs decorator")
        setattr(hook, "_graphion_input_dependencies", dependencies)
        return hook

    return decorate
