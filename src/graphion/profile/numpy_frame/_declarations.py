"""Field specifiers and class-definition-time annotation resolution."""

import inspect
import sys
from dataclasses import dataclass
from typing import Any, Literal, get_origin, get_type_hints, overload

import numpy as np
from numpy.typing import NDArray

from graphion.errors import DeclarationError, StorageError, UnavailablePortError

from . import _dimensions

Array = NDArray[np.generic]


class _Missing:
    __slots__ = ()

    def __repr__(self):
        return "MISSING"


MISSING = _Missing()


@dataclass(frozen=True)
class _Token:
    name: str

    def __repr__(self):
        return self.name


Sample = _Token("Sample")
C, N = _dimensions.C, _dimensions.N


@dataclass(frozen=True)
class Axis:
    name: str


@dataclass(frozen=True, eq=False)
class _Declaration(_dimensions._Parameter):
    kind: str
    dtype: object = None
    axes: tuple | None = None
    default: object = MISSING


@dataclass(frozen=True, slots=True, eq=False)
class _Field(_dimensions._Parameter):
    name: str
    kind: str
    value_type: object
    dtype: object
    axes: tuple | None
    default: object = MISSING

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        try:
            return instance._values[self.name]
        except KeyError:
            raise UnavailablePortError(f"{instance._identity()}.{self.name}: value is not available") from None

    def __set__(self, instance, value):
        instance._assign(self, value)

    def __delete__(self, instance):
        raise StorageError(f"{self.name}: declarations cannot be deleted")


def dimension_bindings(cls, declarations):
    names = {}
    parameters = {}
    for name, declaration in declarations.items():
        if id(declaration) in names:
            raise DeclarationError(f"{cls.__qualname__}: one declaration cannot be bound to two names")
        names[id(declaration)] = name
        if declaration.kind == "Param":
            parameters[id(declaration)] = name
    for base in cls.__mro__[1:]:
        for name, field in vars(base).get("schema", {}).items():
            if isinstance(field, _Field) and field.kind == "Param":
                parameters[id(field)] = name
    return parameters


def bind_axes(axes, parameters, cls):
    if axes is None:
        return None
    bound = []
    for axis in axes:
        if isinstance(axis, _dimensions._Parameter):
            raise DeclarationError("bare declaration dimensions are unsupported; use Axis(name)")
        if isinstance(axis, _dimensions.Expression):
            axis = _dimensions.bind(axis, parameters, cls)
        bound.append(axis)
    return tuple(bound)


def _unsupported(kind, args, kwargs):
    if args or kwargs:
        raise DeclarationError(f"{kind}: unsupported field arguments")


@overload
def Param(*, default: Any) -> Any: ...


@overload
def Param() -> Any: ...


def Param(*args, default=MISSING, **kwargs):
    _unsupported("Param", args, kwargs)
    return _Declaration("Param", default=default)


@overload
def Control(*, default: Any) -> Any: ...


@overload
def Control() -> Any: ...


def Control(*args, default=MISSING, **kwargs):
    _unsupported("Control", args, kwargs)
    return _Declaration("Control", default=default)


# Literal init defaults are standard field-specifier typing metadata. Runtime
# implementations deliberately capture and reject an explicitly supplied init.
@overload
def In(*, dtype: Any = Sample, init: Literal[False] = False) -> Any: ...


@overload
def In(axis: object, /, *axes: object, dtype: Any = Sample, init: Literal[False] = False) -> Any: ...


def In(*axes, dtype=Sample, **kwargs):
    _unsupported("In", (), kwargs)
    return _Declaration("In", dtype, axes)


@overload
def Out(*, dtype: Any = Sample, init: Literal[False] = False) -> Any: ...


@overload
def Out(axis: object, /, *axes: object, dtype: Any = Sample, init: Literal[False] = False) -> Any: ...


def Out(*axes, dtype=Sample, **kwargs):
    _unsupported("Out", (), kwargs)
    return _Declaration("Out", dtype, axes)


@overload
def State(*, default: Any, init: Literal[False] = False) -> Any: ...


@overload
def State(axis: object, /, *axes: object, default: Any, dtype: Any = Sample, init: Literal[False] = False) -> Any: ...


def State(*axes, default=MISSING, dtype=MISSING, **kwargs):
    _unsupported("State", (), kwargs)
    if default is MISSING:
        raise DeclarationError("State: a default is required")
    if not axes:
        if dtype is not MISSING:
            raise DeclarationError("State: scalar State does not accept dtype")
        return _Declaration("State", default=default)
    return _Declaration("State", Sample if dtype is MISSING else dtype, axes, default)


def own_annotations(cls):
    # Explicit annotation dictionaries already contain the intended objects.
    # Preserve Python's resolved objects and the lexical scope of deferred hints.
    # Unresolved ordinary-attribute hints can remain ForwardRef objects.
    if "__annotations__" in vars(cls):
        return dict(vars(cls)["__annotations__"])
    if sys.version_info >= (3, 14):
        import annotationlib

        return annotationlib.get_annotations(cls, format=annotationlib.Format.FORWARDREF)
    return inspect.get_annotations(cls, eval_str=False)


def is_array_configuration(value_type):
    # NumPy also exposes NDArray as a PEP 695 alias on newer interpreters.
    origin = get_origin(value_type)
    return value_type is np.ndarray or value_type is NDArray or origin is np.ndarray or origin is NDArray


def resolve_value_type(cls, name, annotations, declaration):
    label = f"{cls.__qualname__}.{name}"
    if name not in annotations:
        raise DeclarationError(f"{label}: field annotation is required")
    localns = {}
    for base in reversed(cls.__mro__):
        localns.update(vars(base))
    # PEP 695 parameters have their own annotation scope, not class attributes.
    localns.update({parameter.__name__: parameter for parameter in cls.__type_params__})
    localns[cls.__name__] = cls

    # Resolve one annotation at a time, so ordinary attribute hints are irrelevant.
    def holder():
        pass

    holder.__annotations__ = {name: annotations[name]}
    try:
        value_type = get_type_hints(holder, vars(sys.modules[cls.__module__]), localns)[name]
    except Exception as exc:
        raise DeclarationError(f"{label}: cannot resolve field annotation") from exc
    if declaration.kind in ("In", "Out") or declaration.axes:
        if value_type != Array:
            raise DeclarationError(f"{label}: ports and array State require the Array annotation")
    elif declaration.kind == "State":
        if not (
            isinstance(value_type, type) and issubclass(value_type, (bool, int, float, complex, np.number, np.bool_))
        ):
            raise DeclarationError(f"{label}: scalar State requires a numeric or bool annotation")
    return value_type
