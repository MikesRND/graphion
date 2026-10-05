"""Private NumPy operations, not a public backend interface."""

from dataclasses import dataclass

import numpy as np

from graphion.errors import DtypeError, ParamError, ShapeError, StorageError

OVERLAP_MAX_WORK = 1000


@dataclass(frozen=True, slots=True)
class Writer:
    name: str
    array: np.ndarray
    supplied: bool = False


def require_array(array, name):
    if not isinstance(array, np.ndarray):
        raise DtypeError(f"{name}: expected a NumPy ndarray, got {type(array).__name__}")


def is_writable(array):
    if not isinstance(array, np.ndarray):
        return None
    return bool(array.flags.writeable)


def validate(array, shape, dtype, name, *, writable=False):
    require_array(array, name)
    if array.shape != shape:
        raise ShapeError(f"{name}: expected shape {shape}, got {array.shape}")
    if array.dtype != dtype:
        raise DtypeError(f"{name}: expected dtype {dtype}, got {array.dtype}")
    if writable and is_writable(array) is not True:
        raise StorageError(f"{name}: output storage must be known writable")


def allocate(shape, dtype):
    return np.empty(shape, dtype=dtype)


def validate_output_layout(array, name):
    """Conservatively prove that supplied output elements have disjoint bytes."""
    if 0 in array.shape:
        return
    dimensions = sorted((abs(stride), length) for stride, length in zip(array.strides, array.shape) if length > 1)
    span = array.dtype.itemsize
    for stride, length in dimensions:
        if stride < span:
            raise StorageError(
                f"{name}: unsupported output layout: shape={array.shape}, "
                f"strides={array.strides}; use contiguous output storage"
            )
        span += (length - 1) * stride


def readonly_view(array):
    view = array.view()
    view.flags.writeable = False
    return view


def host_copy(value, *, readonly=False):
    array = np.array(value, copy=True)
    if readonly:
        array.flags.writeable = False
    return array


def configuration_array(value, name, *, copy=True, readonly=False):
    """Convert host configuration without allowing shared Python-object storage."""
    try:
        array = np.array(value, copy=True) if copy else np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParamError(f"{name}: cannot convert array configuration") from exc
    if array.dtype.hasobject:
        raise ParamError(f"{name}: array configuration cannot contain object-dtype storage")
    if readonly:
        array.flags.writeable = False
    return array


def initialize(array, default, name):
    if isinstance(default, np.ndarray):
        validate(default, array.shape, array.dtype, name)
        np.copyto(array, default, casting="no")
    elif np.isscalar(default):
        # NumPy complex scalars can silently lose their imaginary part in fill,
        # while Python complex scalars raise. Give both the same dtype error.
        if np.iscomplexobj(default) and array.dtype.kind != "c":
            raise DtypeError(f"{name}: complex State default cannot initialize {array.dtype}")
        try:
            array.fill(default)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DtypeError(f"{name}: State default cannot be converted to {array.dtype}") from exc
    else:
        raise DtypeError(f"{name}: State default must be a scalar or ndarray")


def require_disjoint(a, a_name, b, b_name):
    if not np.may_share_memory(a, b):
        return
    try:
        overlaps = np.shares_memory(a, b, max_work=OVERLAP_MAX_WORK)
    except np.exceptions.TooHardError as exc:
        raise StorageError(f"overlap check exhausted its work limit: {a_name} and {b_name}") from exc
    if overlaps:
        raise StorageError(f"storage overlaps: {a_name} and {b_name}")


def validate_writers(writers):
    for i, writer in enumerate(writers):
        if writer.supplied:
            for j, other in enumerate(writers):
                if i != j:
                    require_disjoint(writer.array, writer.name, other.array, other.name)


def validate_input_storage(array, name, writers):
    for writer in writers:
        require_disjoint(array, name, writer.array, writer.name)
