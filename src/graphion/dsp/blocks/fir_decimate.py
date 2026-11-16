"""A causal FIR decimator with a selectable sampling phase."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from graphion.dsp._validation import is_integer
from graphion.profile.numpy_frame import Array, Axis, Block, C, In, N, Out, Param, SetupError, State

from ._fir import filter_config


class FirDecimate(Block):
    """Causal FIR; output m is filtered input sample m*factor+phase."""

    taps: Array = Param()
    factor: int = Param()
    phase: int = Param(default=0)
    x: Array = In(C, N)
    y: Array = Out(C, N / factor)
    stream_ports = ("x", "y")
    history: Array = State(C, Axis("n_hist"), default=0)

    @property
    def n_hist(self):
        return len(self.taps) - 1

    def setup(self, ctx):
        coeffs = filter_config(ctx, self.taps, self.factor)
        if not is_integer(self.phase, 0) or self.phase >= self.factor:
            raise SetupError("phase must be an integer in [0, factor)")
        self._line = np.zeros((ctx.channels, self.n_hist + ctx.n), ctx.dtype)
        self._previous = self._line[:, : self.n_hist]
        self._current = self._line[:, self.n_hist :]
        self._carry = self._line[:, ctx.n :]
        self._windows = sliding_window_view(self._line, len(coeffs), axis=1)[:, self.phase :: self.factor]
        self._coeffs = coeffs[::-1].copy()

    def process(self):
        self._previous[...] = self.history
        self._current[...] = self.x
        np.matmul(self._windows, self._coeffs, out=self.y)
        self.history[...] = self._carry
