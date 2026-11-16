"""A polyphase FIR interpolator."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from graphion.profile.numpy_frame import Array, Axis, Block, C, In, N, Out, Param, State

from ._fir import filter_config


class FirInterpolate(Block):
    """Polyphase zero-insertion and causal convolution, with no implicit gain."""

    taps: Array = Param()
    factor: int = Param()
    x: Array = In(C, N)
    y: Array = Out(C, N * factor)
    stream_ports = ("x", "y")
    history: Array = State(C, Axis("n_hist"), default=0)

    @property
    def n_hist(self):
        return (len(self.taps) + self.factor - 1) // self.factor - 1

    def setup(self, ctx):
        coeffs = filter_config(ctx, self.taps, self.factor)
        width = self.n_hist + 1
        padded = np.zeros(width * self.factor, ctx.dtype)
        padded[: len(coeffs)] = coeffs
        self._coeffs = padded.reshape(width, self.factor)[::-1].T.copy()
        self._line = np.zeros((ctx.channels, self.n_hist + ctx.n), ctx.dtype)
        self._previous = self._line[:, : self.n_hist]
        self._current = self._line[:, self.n_hist :]
        self._carry = self._line[:, ctx.n :]
        self._windows = sliding_window_view(self._line, width, axis=1)

    def process(self):
        self._previous[...] = self.history
        self._current[...] = self.x
        for phase in range(self.factor):
            np.matmul(self._windows, self._coeffs[phase], out=self.y[:, phase :: self.factor])
        self.history[...] = self._carry
