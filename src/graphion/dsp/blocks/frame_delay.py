"""A whole-frame delay whose split phases support feedback."""

import numpy as np

from graphion.profile.numpy_frame import Array, Block, C, In, N, Out, State, needs_inputs


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
