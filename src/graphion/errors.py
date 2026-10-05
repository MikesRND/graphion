"""Errors raised by Graphion block contracts."""


class BlockError(Exception):
    pass


class DeclarationError(BlockError):
    pass


class ParamError(BlockError):
    pass


class SetupError(BlockError):
    pass


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
