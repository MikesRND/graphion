"""Continuous pseudorandom binary sources."""

import numpy as np

from graphion.profile.numpy_frame import Array, Axis, Block, C, N, Out, Param, SetupError, State


class Prbs(Block):
    """Emit MSB then shift left, inserting MSB XOR bit q-1.

    Non-inverted by default. ITU-T O.150 inverts orders 15, 23 and 31.
    State holds only the next order bits, independent of the sequence period.
    """

    order: int = Param(default=15)
    seed: object = Param(default=None)
    invert: bool = Param(default=False)
    y: Array = Out(C, N, dtype=np.uint8)
    stream_ports = ("y",)
    register: Array = State(C, Axis("order"), dtype=np.uint8, default=0)

    def setup(self, ctx):
        taps = {7: 6, 9: 5, 15: 14, 23: 18, 31: 28}
        if isinstance(self.order, bool) or not isinstance(self.order, (int, np.integer)) or self.order not in taps:
            raise SetupError("PRBS order must be one of 7, 9, 15, 23, 31")
        self._q = taps[self.order]
        seeds = np.asarray((1 << self.order) - 1 if self.seed is None else self.seed)
        if seeds.dtype.kind not in "iu" or seeds.shape not in ((), (ctx.channels,)):
            raise SetupError("seed must be an integer or one integer per channel")
        if np.any(seeds < 1) or np.any(seeds >= 1 << self.order):
            raise SetupError("seed must be nonzero and fit in order bits")
        seeds = np.broadcast_to(seeds, (ctx.channels,)).astype(np.uint64)
        shifts = np.arange(self.order - 1, -1, -1, dtype=np.uint64)
        self._initial = ((seeds[:, None] >> shifts) & 1).astype(np.uint8)
        self._line = np.empty((ctx.channels, self.order + ctx.n), dtype=np.uint8)

    def reset(self):
        self.register[...] = self._initial

    def process(self):
        n, order, q = self.ctx.n, self.order, self._q
        self._line[:, :order] = self.register
        for start in range(order, order + n, q):
            size = min(q, order + n - start)
            np.bitwise_xor(
                self._line[:, start - order : start - order + size],
                self._line[:, start - q : start - q + size],
                out=self._line[:, start : start + size],
            )
        self.y[...] = self._line[:, :n]
        self.register[...] = self._line[:, n : n + order]
        if self.invert:
            np.bitwise_xor(self.y, 1, out=self.y)
