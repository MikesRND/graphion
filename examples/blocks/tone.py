"""A real sine-wave source with continuous phase across frames."""

import numpy as np

from graphion import Array, Block, C, Control, Ctx, N, Out, SetupError, State


class Tone(Block):
    freq: float = Control(default=10.0)
    amplitude: float = Control(default=1.0)
    y: Array = Out(C, N)
    phase: float = State(default=0.0)

    def setup(self, ctx: Ctx):
        if self.ctx.dtype.kind != "f":
            raise SetupError("Tone needs real floating samples")
        self._k = np.arange(ctx.n)

    def process(self):
        w = 2 * np.pi * self.freq / float(self.ctx.rate)
        self.y[...] = self.amplitude * np.sin(self.phase + w * self._k)
        self.phase = (self.phase + w * self.ctx.n) % (2 * np.pi)
