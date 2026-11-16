"""Errors raised by Graphion block contracts."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .profile.numpy_frame.block import Block


class BlockError(Exception):
    pass


class DeclarationError(BlockError):
    pass


class ParamError(BlockError):
    pass


class SetupError(BlockError):
    _context_block: "Block"


class ExecutionError(BlockError):
    pass


class RegistrationError(BlockError):
    pass


class WiringError(BlockError):
    pass


class StorageError(BlockError):
    pass


class UnavailablePortError(BlockError):
    pass


class ShapeError(BlockError):
    pass


class DtypeError(BlockError):
    pass
