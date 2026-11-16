"""NumPy-only FIR coefficient design."""

import numpy as np

from graphion.dsp._validation import is_integer
from graphion.profile.numpy_frame import Array


def lowpass(ntaps: int, cutoff: float, beta: float = 8.0) -> Array:
    """Return unity-DC-gain taps; cutoff is in input cycles/sample."""
    if ntaps < 1 or ntaps % 2 == 0 or not 0 < cutoff < 0.5:
        raise ValueError("lowpass needs an odd positive tap count and cutoff in (0, 0.5)")
    k = np.arange(ntaps) - (ntaps - 1) / 2
    taps = 2 * cutoff * np.sinc(2 * cutoff * k) * np.kaiser(ntaps, beta)
    return taps / taps.sum()


def rrc(span: int, sps: int, rolloff: float) -> Array:
    """Return span*sps+1 symmetric, unit-energy root-raised-cosine taps.

    At sps=1, nonzero rolloff aliases the intended continuous-time pulse.
    Integer sps>=2 avoids that aliasing; finite span can still leave residual
    inter-symbol interference in the matched pair. The conjugate-reversed
    discrete filter remains a matched filter at every supported sps.
    """
    for name, value in (("span", span), ("sps", sps)):
        if not is_integer(value, 1):
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(rolloff) or not 0 <= rolloff <= 1:
        raise ValueError("rolloff must lie in [0, 1]")
    t = (np.arange(span * sps + 1, dtype=np.float64) - span * sps / 2) / sps
    if rolloff == 0:
        h = np.sinc(t)
    else:
        beta = rolloff
        zero = t == 0
        singular = np.isclose(np.abs(4 * beta * t), 1, rtol=0, atol=1e-14)
        regular = ~(zero | singular)
        u = t[regular]
        h = np.empty_like(t)
        h[regular] = (np.sin(np.pi * u * (1 - beta)) + 4 * beta * u * np.cos(np.pi * u * (1 + beta))) / (
            np.pi * u * (1 - (4 * beta * u) ** 2)
        )
        h[zero] = 1 + beta * (4 / np.pi - 1)
        h[singular] = (
            beta
            / np.sqrt(2)
            * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        )
    return h / np.linalg.norm(h)
