"""Errors raised by Graphion block contracts."""


class BlockError(Exception):
    """Base class for Graphion contract errors."""


class DeclarationError(BlockError):
    """Invalid or conflicting block declaration, hook, or field specifier."""


class ParamError(BlockError):
    """Missing, unknown, invalid, or forbidden configuration."""


class SetupError(BlockError):
    """Invalid context, lifecycle call, or setup state."""

    # Only framework ctx lookup errors set this marker, so most instances lack
    # it. Profiles test for its presence and never read it, which keeps this
    # profile-agnostic module free of profile types.
    _context_block: object


class ExecutionError(BlockError):
    """Invalid processing entry point, phase sequence, or hook result."""


class RegistrationError(BlockError):
    """Invalid child registration or lifecycle relationship."""


class WiringError(BlockError):
    """Invalid connection or assignment to a connected input."""


class StorageError(BlockError):
    """Invalid storage replacement, binding, layout, or overlap."""


class UnavailablePortError(BlockError):
    """An input or array buffer has no available value."""


class ShapeError(BlockError):
    """An array shape does not match its resolved declaration."""


class DtypeError(BlockError):
    """An array value or dtype does not satisfy its declaration."""
