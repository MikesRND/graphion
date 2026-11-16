"""Shared fixtures for the per-block DSP library tests."""

import numpy as np

from graphion.dsp.communications import bpsk, qpsk
from graphion.profile.numpy_frame import Ctx


def context(n=64, channels=1, dtype=np.float64):
    return Ctx(rate=1, n=n, channels=channels, dtype=dtype)


def constellation_tables():
    """BPSK, QPSK and a rectangular 16-point table."""
    return [bpsk(), qpsk(), np.array([i + 1j * q for i in (-3, -1, 1, 3) for q in (-3, -1, 1, 3)])]


def label_bits(table):
    """Every table index as MSB-first unpacked bits, in index order, shaped (1, bits)."""
    k = len(table).bit_length() - 1
    return ((np.arange(len(table))[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).reshape(1, -1)
