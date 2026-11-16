"""Cumulative, per-channel element error measurement."""

import numpy as np

from graphion.profile.numpy_frame import Array, Block, C, In, N, Out, Param, SetupError, State


class ErrorCounter(Block):
    """Count unequal elements after excluding skip leading elements per channel."""

    skip: int = Param(default=0)
    x: Array = In(C, N)
    ref: Array = In(C, N)
    total: Array = Out(C, dtype=np.uint64)
    errors: Array = Out(C, dtype=np.uint64)
    rate: Array = Out(C, dtype=np.float64)
    stream_ports = ("x", "ref")
    compared: Array = State(C, dtype=np.uint64, default=0)
    unequal: Array = State(C, dtype=np.uint64, default=0)
    remaining: int = State(default=0)

    def setup(self, ctx):
        if isinstance(self.skip, bool) or not isinstance(self.skip, (int, np.integer)) or self.skip < 0:
            raise SetupError("skip must be a nonnegative integer")
        self._mask = np.empty((ctx.channels, ctx.n), dtype=bool)
        self._weights = np.empty((ctx.channels, ctx.n), dtype=np.uint64)
        self._counts = np.empty(ctx.channels, dtype=np.uint64)

    def reset(self):
        self.remaining = self.skip

    def process(self):
        start = min(self.remaining, self.ctx.n)
        self.remaining -= start
        np.not_equal(self.x[:, start:], self.ref[:, start:], out=self._mask[:, start:])
        self._weights[:, start:] = self._mask[:, start:]
        np.sum(self._weights[:, start:], axis=1, out=self._counts)
        np.add(self.unequal, self._counts, out=self.unequal)
        np.add(self.compared, self.ctx.n - start, out=self.compared)
        self.total[...] = self.compared
        self.errors[...] = self.unequal
        if self.compared[0]:
            np.divide(self.unequal, self.compared, out=self.rate)
        else:
            self.rate.fill(np.nan)
