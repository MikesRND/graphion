"""MSB-first table mapping and bounded-memory hard symbol decisions."""

import numpy as np

from graphion.profile.numpy_frame import Array, Axis, Block, C, In, N, Out, Param, SetupError


def _constellation(value):
    table = np.asarray(value, dtype=np.complex128)
    if table.ndim != 1 or len(table) < 2 or len(table) & (len(table) - 1):
        raise SetupError("constellation must be a vector of power-of-two length >= 2")
    if not np.isfinite(table).all() or len(np.unique(table)) != len(table):
        raise SetupError("constellation points must be finite and distinct")
    return table.copy(), len(table).bit_length() - 1


class SymbolMapper(Block):
    """Map unpacked binary groups to supplied points; output is complex128."""

    constellation: Array = Param()
    x: Array = In(C, N, dtype=np.uint8)
    y: Array = Out(C, Axis("symbols"), dtype=np.complex128)
    stream_ports = ("x", "y")

    @property
    def symbols(self):
        return self.ctx.n // self._k

    def setup(self, ctx):
        self._table, self._k = _constellation(self.constellation)
        if ctx.n % self._k:
            raise SetupError("bit frame length must be divisible by bits per symbol")
        self._index = np.empty((ctx.channels, self.symbols), dtype=np.intp)
        self._bit = np.empty_like(self._index)
        self._invalid = np.empty((ctx.channels, ctx.n), dtype=bool)

    def process(self):
        np.greater(self.x, 1, out=self._invalid)
        if self._invalid.any():
            raise ValueError("SymbolMapper requires unpacked bits with values 0 or 1")
        self._index.fill(0)
        for bit in range(self._k):
            np.left_shift(self._index, 1, out=self._index)
            self._bit[...] = self.x[:, bit :: self._k]
            np.bitwise_or(self._index, self._bit, out=self._index)
        # clip avoids np.take's buffered-output temporary in its default raise mode.
        np.take(self._table, self._index, out=self.y, mode="clip")


class SymbolDemapper(Block):
    """Nearest point, ties to lowest index, then MSB-first unpacked bits.

    Workspaces scale with frame size, not constellation size. Iterating points
    trades speed on medium constellations for bounded memory on large frames.
    """

    constellation: Array = Param()
    x: Array = In(C, N)
    y: Array = Out(C, Axis("bits"), dtype=np.uint8)
    stream_ports = ("x", "y")

    @property
    def bits(self):
        return self.ctx.n * self._k

    def setup(self, ctx):
        self._table, self._k = _constellation(self.constellation)
        if ctx.dtype not in (np.complex64, np.complex128):
            raise SetupError("SymbolDemapper needs complex64 or complex128 samples")
        shape = (ctx.channels, ctx.n)
        # Double precision also preserves distinct supplied points for complex64 input.
        self._delta = np.empty(shape, dtype=np.complex128)
        self._metric = np.empty(shape, dtype=np.float64)
        self._best = np.empty(shape, dtype=np.float64)
        self._mask = np.empty(shape, dtype=bool)
        self._index = np.empty(shape, dtype=np.intp)
        self._bits = np.empty(shape, dtype=np.intp)

    def process(self):
        self._best.fill(np.inf)
        self._index.fill(0)
        for index, point in enumerate(self._table):
            np.subtract(self.x, point, out=self._delta)
            np.absolute(self._delta, out=self._metric)
            np.less(self._metric, self._best, out=self._mask)
            np.copyto(self._best, self._metric, where=self._mask)
            np.copyto(self._index, index, where=self._mask)
        for bit in range(self._k):
            np.right_shift(self._index, self._k - bit - 1, out=self._bits)
            np.bitwise_and(self._bits, 1, out=self._bits)
            self.y[:, bit :: self._k] = self._bits
