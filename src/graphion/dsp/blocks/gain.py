"""Multiplication by a controllable gain."""

import numpy as np

from graphion.profile.numpy_frame import Array, Block, C, Control, In, N, Out


class Gain(Block):
    """Multiply each input element by the gain control."""

    gain: float = Control(default=1.0)
    x: Array = In(C, N)
    y: Array = Out(C, N)
    stream_ports = ("x", "y")

    def output(self):
        np.multiply(self.x, self.gain, out=self.y)
