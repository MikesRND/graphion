"""Configuration checks shared within the DSP library."""

import numpy as np


def is_integer(value, minimum=None):
    """Whether value is a built-in or NumPy integer, never a bool, and at least minimum."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return False
    return minimum is None or value >= minimum
