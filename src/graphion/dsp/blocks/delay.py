"""Delay blocks for the Python/NumPy frame Implementation Profile."""

import numpy as np

from graphion.profile.numpy_frame import Array, Axis, Block, C, In, N, Out, Param, SetupError, State, needs_inputs


class FrameDelay(Block):
    x: Array = In(C, N)
    y: Array = Out(C, N)
    stream_ports = ("x", "y")
    held: Array = State(C, N, default=0.0)

    @needs_inputs(None)
    def output(self):
        np.copyto(self.y, self.held, casting="no")

    def update(self):
        np.copyto(self.held, self.x, casting="no")


class SampleDelay(Block):
    """A zero-initialized causal delay, including delays longer than a frame."""

    delay: int = Param()
    x: Array = In(C, N)
    y: Array = Out(C, N)
    stream_ports = ("x", "y")
    history: Array = State(C, Axis("delay"), default=0)

    def setup(self, ctx):
        if isinstance(self.delay, bool) or not isinstance(self.delay, (int, np.integer)) or self.delay < 0:
            raise SetupError("delay must be a nonnegative integer")
        self._line = np.empty((ctx.channels, self.delay + ctx.n), ctx.dtype)

    def process(self):
        self._line[:, : self.delay] = self.history
        self._line[:, self.delay :] = self.x
        self.y[...] = self._line[:, : self.ctx.n]
        self.history[...] = self._line[:, self.ctx.n :]
