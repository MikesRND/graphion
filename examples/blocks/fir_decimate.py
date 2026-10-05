"""Windowed-sinc coefficients and a streaming FIR decimator."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from graphion import Array, Axis, Block, C, Ctx, In, N, Out, Param, SetupError, State


def lowpass(ntaps: int, cutoff: float, beta: float = 8.0) -> Array:
    """Return unity-DC-gain taps; cutoff is in input cycles/sample."""
    if ntaps < 1 or ntaps % 2 == 0 or not 0 < cutoff < 0.5:
        raise ValueError("lowpass needs an odd positive tap count and cutoff in (0, 0.5)")
    k = np.arange(ntaps) - (ntaps - 1) / 2
    taps = 2 * cutoff * np.sinc(2 * cutoff * k) * np.kaiser(ntaps, beta)
    return taps / taps.sum()


class FirDecimate(Block):
    """Filter with real taps, then keep every `factor`-th sample.

    Samples are float64 or complex128 and taps are float64; setup rejects
    other dtypes. Processing reuses storage allocated in setup, avoiding
    frame-sized temporaries on each call.
    """

    taps: Array = Param()
    factor: int = Param()
    x: Array = In(C, N)
    y: Array = Out(C, N / factor)
    history: Array = State(C, Axis("n_hist"), default=0)

    @property
    def n_hist(self):
        return len(self.taps) - 1

    def setup(self, ctx: Ctx):
        if isinstance(self.factor, bool) or not isinstance(self.factor, (int, np.integer)) or self.factor <= 0:
            raise SetupError("factor must be a positive integer")
        if self.taps.ndim != 1 or self.taps.size == 0 or not np.isfinite(self.taps).all():
            raise SetupError("taps must be a nonempty finite vector")
        if ctx.dtype not in (np.float64, np.complex128) or self.taps.dtype != np.float64:
            raise SetupError("FirDecimate needs float64 or complex128 samples and float64 taps")
        # Delay line: the tail of the previous frame, followed by the current frame.
        self._line = np.zeros((ctx.channels, self.n_hist + ctx.n), ctx.dtype)
        self._previous = self._line[:, : self.n_hist]
        self._current = self._line[:, self.n_hist :]
        self._carry = self._line[:, ctx.n :]  # what the next call needs
        # Row m holds the len(taps) samples that produce output m.
        self._windows = sliding_window_view(self._line, len(self.taps), axis=1)[:, :: self.factor]
        self._coeffs = self.taps[::-1].astype(ctx.dtype)  # exact for float64 -> complex128

    def process(self):
        self._previous[...] = self.history
        self._current[...] = self.x
        np.matmul(self._windows, self._coeffs, out=self.y)
        self.history[...] = self._carry
