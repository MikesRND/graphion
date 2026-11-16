"""MSB-first mapping of unpacked bits to constellation points."""

import numpy as np

from graphion.profile.numpy_frame import Array, Axis, Block, C, In, N, Out, Param, SetupError

from ._constellation import constellation_table


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
        self._table, self._k = constellation_table(self.constellation)
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
