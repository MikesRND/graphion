"""Shared configuration checks for the FIR rate-changing blocks."""

import numpy as np

from graphion.dsp._validation import is_integer
from graphion.profile.numpy_frame import SetupError


def filter_config(ctx, taps, factor):
    if not is_integer(factor, 1):
        raise SetupError("factor must be a positive integer")
    taps = np.asarray(taps)
    if taps.ndim != 1 or taps.size == 0 or not np.isfinite(taps).all():
        raise SetupError("taps must be a nonempty finite vector")
    if ctx.dtype not in (np.float32, np.float64, np.complex64, np.complex128):
        raise SetupError("FIR filters need float32/float64/complex64/complex128 samples")
    if ctx.dtype.kind == "f" and np.iscomplexobj(taps):
        raise SetupError("complex taps require complex samples")
    coeffs = taps.astype(ctx.dtype)
    if not np.isfinite(coeffs).all():
        raise SetupError("taps must be finite at the stream precision")
    return coeffs
