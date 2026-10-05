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
    "Array",
    "Axis",
    "Block",
    "needs_inputs",
    "Ctx",
    "PortSpec",
    "MISSING",
    "Param",
    "Control",
    "State",
    "In",
    "Out",
    "Sample",
    "C",
    "N",
    "BlockError",
    "DeclarationError",
    "ParamError",
    "SetupError",
    "ExecutionError",
    "RegistrationError",
    "WiringError",
    "StorageError",
    "UnavailablePortError",
    "ShapeError",
    "DtypeError",
]
