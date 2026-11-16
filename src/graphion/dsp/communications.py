"""Constellations, nominal energy conversions and coherent AWGN BER."""

import math

import numpy as np


def bpsk():
    """Unit-energy BPSK, 0 -> +1 and 1 -> -1."""
    return np.array([1, -1], dtype=np.complex128)


def qpsk():
    """Unit-energy Gray QPSK: MSB selects I sign, LSB selects Q sign."""
    return np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def ebn0_to_snr_db(ebn0_db, bits_per_symbol, samples_per_symbol, *, real=False):
    """Convert Eb/N0 dB to nominal signal power / total sample noise power dB."""
    _positive(bits_per_symbol, "bits_per_symbol")
    _positive(samples_per_symbol, "samples_per_symbol")
    return np.asarray(ebn0_db) + 10 * np.log10(bits_per_symbol / samples_per_symbol) + (10 * np.log10(2) if real else 0)


def esn0_to_snr_db(esn0_db, samples_per_symbol, *, real=False):
    """Convert Es/N0 dB to sample SNR dB."""
    return ebn0_to_snr_db(esn0_db, 1, samples_per_symbol, real=real)


def bpsk_qpsk_ber(ebn0_db):
    """Theoretical coherent BPSK / Gray QPSK BER; scalar or array input."""
    values = np.asarray(ebn0_db, dtype=np.float64)
    gamma = np.power(10.0, values / 10)
    result = np.fromiter((0.5 * math.erfc(math.sqrt(g)) for g in gamma.flat), dtype=np.float64).reshape(values.shape)
    return float(result) if result.ndim == 0 else result
