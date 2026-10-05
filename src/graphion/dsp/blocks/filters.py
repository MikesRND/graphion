"""Filters for the Python/NumPy frame Implementation Profile."""

import numpy as np

from graphion.profile.numpy_frame import Array, Block, C, In, N, Out, Param, SetupError, State


class OnePoleLowpass(Block):
    pole: float = Param(default=0.9)
    x: Array = In(C, N)
    y: Array = Out(C, N)
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
