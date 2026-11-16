"""Shared constellation-table validation for the symbol mapping blocks."""

import numpy as np

from graphion.profile.numpy_frame import SetupError


def constellation_table(value):
    table = np.asarray(value, dtype=np.complex128)
    if table.ndim != 1 or len(table) < 2 or len(table) & (len(table) - 1):
        raise SetupError("constellation must be a vector of power-of-two length >= 2")
    if not np.isfinite(table).all() or len(np.unique(table)) != len(table):
        raise SetupError("constellation points must be finite and distinct")
    return table.copy(), len(table).bit_length() - 1
