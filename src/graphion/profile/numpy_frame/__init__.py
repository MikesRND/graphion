"""Authoring API for the synchronous Python/NumPy frame Implementation Profile."""

from graphion.errors import (
    BlockError,
    DeclarationError,
    DtypeError,
    ExecutionError,
    ParamError,
    RegistrationError,
    SetupError,
    ShapeError,
    StorageError,
    UnavailablePortError,
    WiringError,
)

from ._declarations import MISSING, Array, Axis, C, Control, In, N, Out, Param, Sample, State
from ._dependencies import needs_inputs
from .block import Block
from .context import Ctx
from .spec import PortSpec

__all__ = [
    "MISSING",
    "Array",
    "Axis",
    "Block",
    "BlockError",
    "C",
    "Control",
    "Ctx",
    "DeclarationError",
    "DtypeError",
    "ExecutionError",
    "In",
    "N",
    "Out",
    "Param",
    "ParamError",
    "PortSpec",
    "RegistrationError",
    "Sample",
    "SetupError",
    "ShapeError",
    "State",
    "StorageError",
    "UnavailablePortError",
    "WiringError",
    "needs_inputs",
]
