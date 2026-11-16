"""A sample delay of any nonnegative length."""

import numpy as np

from graphion.dsp._validation import is_integer
from graphion.profile.numpy_frame import Array, Axis, Block, C, In, N, Out, Param, SetupError, State


class SampleDelay(Block):
    """A zero-initialized causal delay, including delays longer than a frame."""

    delay: int = Param()
    x: Array = In(C, N)
    y: Array = Out(C, N)
    stream_ports = ("x", "y")
    history: Array = State(C, Axis("delay"), default=0)

    def setup(self, ctx):
        if not is_integer(self.delay, 0):
            raise SetupError("delay must be a nonnegative integer")
        self._line = np.empty((ctx.channels, self.delay + ctx.n), ctx.dtype)

    def process(self):
        self._line[:, : self.delay] = self.history
        self._line[:, self.delay :] = self.x
        self.y[...] = self._line[:, : self.ctx.n]
        self.history[...] = self._line[:, self.ctx.n :]
