"""Fixed-storage capture of recent complex samples for display."""

import numpy as np

from graphion.dsp._validation import is_integer
from graphion.profile.numpy_frame import Array, Axis, Block, C, In, N, Out, Param, SetupError, State


class ConstellationViewer(Block):
    """Retain the most recent depth complex samples per channel, oldest first, for display.

    The block captures; it does not draw. history is the retained record,
    and a host may read it between runs: an entry is NaN until a sample has
    been captured for it, so it is all NaN after setup and after reset. points
    mirrors history after each run and, like any output, is unspecified
    before the first run and after reset.
    Storage is fixed at setup: about C * (3 * depth + N) samples.
    """

    depth: int = Param(default=1024)
    skip: int = Param(default=0)
    x: Array = In(C, N)
    points: Array = Out(C, Axis("depth"))
    stream_ports = ("x",)
    history: Array = State(C, Axis("depth"), default=np.nan)
    remaining: int = State(default=0)

    def setup(self, ctx):
        if ctx.dtype not in (np.complex64, np.complex128):
            raise SetupError("ConstellationViewer needs complex64 or complex128 samples")
        if not is_integer(self.depth, 1):
            raise SetupError("depth must be a positive integer")
        if not is_integer(self.skip, 0):
            raise SetupError("skip must be a nonnegative integer")
        self._line = np.empty((ctx.channels, self.depth + ctx.n), ctx.dtype)

    def reset(self):
        self.remaining = self.skip

    def process(self):
        start = min(self.remaining, self.ctx.n)
        self.remaining -= start
        accepted = self.ctx.n - start
        self._line[:, : self.depth] = self.history
        self._line[:, self.depth : self.depth + accepted] = self.x[:, start:]
        # Only the written part of the line is read back.
        self.history[...] = self._line[:, accepted : accepted + self.depth]
        self.points[...] = self.history
