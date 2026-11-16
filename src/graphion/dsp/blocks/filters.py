"""Filters for the Python/NumPy frame Implementation Profile."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from graphion.profile.numpy_frame import Array, Axis, Block, C, In, N, Out, Param, SetupError, State


class OnePoleLowpass(Block):
    pole: float = Param(default=0.9)
    x: Array = In(C, N)
    y: Array = Out(C, N)
    stream_ports = ("x", "y")
    memory: Array = State(C, default=0.0)

    def setup(self, ctx):
        if ctx.dtype.kind not in "fc" or not 0 <= self.pole < 1:
            raise SetupError("OnePoleLowpass needs floating/complex samples and pole in [0, 1)")
        self._feed = 1.0 - self.pole
        self._work = np.empty(ctx.channels, dtype=ctx.dtype)

    def process(self):
        x, y = self.x, self.y
        memory, work = self.memory, self._work
        for i in range(x.shape[1]):
            np.multiply(memory, self.pole, out=memory)
            np.multiply(x[:, i], self._feed, out=work)
            np.add(memory, work, out=memory)
            y[:, i] = memory


def _filter_config(ctx, taps, factor):
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)) or factor <= 0:
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
        coeffs = _filter_config(ctx, self.taps, self.factor)
        if (
            isinstance(self.phase, bool)
            or not isinstance(self.phase, (int, np.integer))
            or not 0 <= self.phase < self.factor
        ):
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
        coeffs = _filter_config(ctx, self.taps, self.factor)
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
