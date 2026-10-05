"""A phase-continuous mixer whose frequency can change between frames."""

import numpy as np

from graphion import Array, Block, C, Control, Ctx, In, N, Out, State


class Tuner(Block):
    freq: float = Control()
    x: Array = In(C, N)
    y: Array = Out(C, N, dtype=np.complex128)
    phase: float = State(default=0.0)

    def setup(self, ctx: Ctx):
        self._k = np.arange(ctx.n)

    def process(self):
        w = 2 * np.pi * self.freq / float(self.ctx.rate)
        self.y[...] = self.x * np.exp(-1j * (self.phase + w * self._k))
        self.phase = (self.phase + w * self.ctx.n) % (2 * np.pi)
