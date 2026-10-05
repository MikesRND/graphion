"""Plain-data inspection snapshots for the Python/NumPy Implementation Profile."""

import math
from copy import deepcopy
from fractions import Fraction
from numbers import Integral
from types import GenericAlias

import numpy as np

from . import _dimensions
from ._declarations import MISSING, Array, Axis, C, N, Sample

_NUMPY_SCALAR_TYPES = tuple(set(np.sctypeDict.values()))


def _type_name(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _number(value):
    value = float(value)
    if math.isfinite(value):
        return value
    return {"float": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}


def _literal(value):
    if isinstance(value, np.generic):
        return MISSING
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, complex):
        return {"real": _number(value.real), "imag": _number(value.imag)}
    if isinstance(value, str):
        return str(value)
    return MISSING


def _value(value):
    if type(value) is Fraction:
        return {"kind": "fraction", "numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, np.ndarray):
        return {"kind": "summary", "dtype": value.dtype.str, "shape": [int(size) for size in value.shape]}
    if isinstance(value, np.generic):
        literal = _literal(value.item())
        if literal is not MISSING:
            return {"kind": "literal", "value": literal, "dtype": value.dtype.str}
    else:
        literal = _literal(value)
        if literal is not MISSING:
            return {"kind": "literal", "value": literal}
    return {"kind": "opaque", "type": _type_name(value)}


def _same_alias(value, expected):
    # Alias equality delegates to its arguments, which can be author objects.
    if value is expected:
        return True
    if type(value) is not GenericAlias or type(expected) is not GenericAlias:
        return False
    return (
        value.__origin__ is expected.__origin__
        and len(value.__args__) == len(expected.__args__)
        and all(_same_alias(a, b) for a, b in zip(value.__args__, expected.__args__))
    )


def _value_type(value):
    if _same_alias(value, Array):
        return {"kind": "array"}
    if isinstance(value, type):
        return {"kind": "named", "module": value.__module__, "qualname": value.__qualname__}
    return {"kind": "opaque", "type": _type_name(value)}


def _resolved(value):
    return {"kind": "resolved", "value": value}


def _invalid(value):
    return {"kind": "invalid", "type": _type_name(value)}


def _safe_dtype_selector(value, active=None):
    """Exclude selector objects that NumPy could inspect or repr as author code."""
    cls = type(value)
    if value is None or any(cls is primitive for primitive in (str, bytes, bool, int, float, complex)):
        return True
    if isinstance(value, np.dtype):
        return True
    if isinstance(value, np.generic):
        return any(cls is scalar for scalar in _NUMPY_SCALAR_TYPES)
    if cls is type:
        return any(value is primitive for primitive in (bool, int, float, complex, str, bytes, object)) or any(
            value is scalar for scalar in _NUMPY_SCALAR_TYPES
        )
    if not any(cls is container for container in (tuple, list, dict)):
        return False
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        return False
    active.add(identity)
    try:
        items = (item for pair in value.items() for item in pair) if cls is dict else value
        return all(_safe_dtype_selector(item, active) for item in items)
    finally:
        active.remove(identity)


def _dtype(selector):
    if selector is Sample:
        return {"kind": "symbol", "name": "Sample"}
    if not _safe_dtype_selector(selector):
        return _invalid(selector)
    try:
        dtype = np.dtype(np.int64 if selector is int else selector)
    except (TypeError, ValueError, OverflowError):
        return _invalid(selector)
    if dtype.kind not in "biufc":
        return _invalid(selector)
    return _resolved(dtype.str)


def _dimension(axis):
    if isinstance(axis, _dimensions.Expression):
        return _dimensions.describe(axis)
    if axis is C or axis is N:
        return {"kind": "symbol", "name": axis.name}
    if isinstance(axis, Axis):
        if isinstance(axis.name, str):
            return {"kind": "axis", "name": str(axis.name)}
        return _invalid(axis)
    if isinstance(axis, Integral) and not isinstance(axis, (bool, np.bool_)) and axis >= 0:
        return _resolved(int(axis))
    return _invalid(axis)


def _storage(block):
    roles = {}
    for name, field in block.schema.items():
        if field.kind == "In":
            roles[name] = "connected_input" if name in block._incoming else "external_input"
        elif field.kind == "Out":
            if name in block._incoming:
                roles[name] = "forwarded_output"
            elif name in block._supplied:
                roles[name] = "supplied_output"
            else:
                roles[name] = "direct_output"
        elif field.kind == "State":
            roles[name] = "array_state" if field.axes else "scalar_state"
    return roles


def _context(block):
    ctx = block._ctx
    if isinstance(ctx.rate, Fraction):
        exact = {"numerator": int(ctx.rate.numerator), "denominator": int(ctx.rate.denominator)}
    elif isinstance(ctx.rate, int):
        exact = {"numerator": ctx.rate, "denominator": 1}
    else:
        exact = None
    # Ctx validates the numeric approximation before the block is configured.
    return {
        "rate": float(ctx.rate),
        "rate_exact": exact,
        "n": int(ctx.n),
        "channels": int(ctx.channels),
        "dtype": ctx.dtype.str,
        "xp": "numpy",
    }


def _declaration(block, field):
    dtype, shape = None, None
    declared_shape = None if field.axes is None else [_dimension(axis) for axis in field.axes]
    if field.axes is not None:
        if block._local_setup_complete:
            resolved_shape, resolved_dtype = block._specs[field.name]
            dtype = _resolved(resolved_dtype.str)
            shape = [_resolved(int(size)) for size in resolved_shape]
        else:
            dtype = _dtype(field.dtype)
            shape = deepcopy(declared_shape)
    if field.kind in ("In", "Out"):
        default = None
    elif field.default is MISSING:
        default = {"kind": "required"}
    else:
        default = _value(field.default)
    return {
        "name": field.name,
        "kind": field.kind,
        "value_type": _value_type(field.value_type),
        "dtype": dtype,
        "shape": shape,
        "declared_shape": declared_shape,
        "default": default,
    }


def describe(block):
    """Describe the selected subtree without block hooks or axis evaluation."""
    nodes = []

    def visit(node, path):
        nodes.append((node, path))
        for name, child in node._children.items():
            visit(child, path + "/" + name.replace("~", "~0").replace("/", "~1"))

    visit(block, "root")
    paths = {id(node): path for node, path in nodes}
    complete = block._tree.ready
    result = {
        "schema_version": 1,
        "profile": "python.numpy.frame",
        "topology": "complete" if complete else "open",
        "blocks": [],
        "connections": [],
        "unresolved": [],
    }

    def unresolved(path, declaration, fact, reason):
        result["unresolved"].append({"path": path, "declaration": declaration, "fact": fact, "reason": reason})

    if not complete:
        unresolved("root", None, "topology", "Lifecycle root setup has not succeeded.")
    for node, path in nodes:
        declarations = [_declaration(node, field) for field in node.schema.values()]
        cls = type(node)
        result["blocks"].append(
            {
                "path": path,
                "definition": {"module": cls.__module__, "qualname": cls.__qualname__},
                "execution_ownership": "opaque",
                "phases": {name: list(inputs) for name, inputs in cls._phase_dependencies.items()},
                "parameters": {
                    name: _value(node._values[name]) for name, field in node.schema.items() if field.kind == "Param"
                },
                "controls": {
                    name: _value(node._values[name]) for name, field in node.schema.items() if field.kind == "Control"
                },
                "declarations": declarations,
                "children": [paths[id(child)] for child in node._children.values()],
                "context": _context(node) if node._local_setup_complete else None,
                "storage": _storage(node) if complete else None,
            }
        )
        if not node._local_setup_complete:
            unresolved(path, None, "context", "Local setup has not succeeded.")
        if not complete:
            unresolved(path, None, "storage", "Lifecycle root setup has not succeeded.")
        for declaration in declarations:
            name = declaration["name"]
            if declaration["value_type"]["kind"] == "opaque":
                unresolved(path, name, "value_type", "Annotation has no portable named type representation.")
            dtype, shape = declaration["dtype"], declaration["shape"]
            if dtype is not None and dtype["kind"] != "resolved":
                unresolved(path, name, "dtype", "Declared dtype is symbolic or invalid.")
            if shape is not None and any(axis["kind"] != "resolved" for axis in shape):
                unresolved(path, name, "shape", "Declared dimensions are symbolic or invalid.")
        for edge in node._connections:
            result["connections"].append(
                {
                    "owner": path,
                    "source": {"path": paths[id(edge.source_block)], "port": str(edge.source_port)},
                    "destination": {"path": paths[id(edge.destination_block)], "port": str(edge.destination_port)},
                }
            )
    return result
