"""Type-checker escape hatch for runtime rejection tests."""

from typing import Any


def misuse(value: object) -> Any:
    """Pass through a deliberate API misuse for runtime validation."""
    return value
