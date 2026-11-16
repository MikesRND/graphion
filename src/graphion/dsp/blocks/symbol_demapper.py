"""Bounded-memory hard nearest-point symbol decisions."""

import numpy as np

from graphion.profile.numpy_frame import Array, Axis, Block, C, In, N, Out, Param, SetupError

from ._constellation import constellation_table


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
        self._table, self._k = constellation_table(self.constellation)
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
