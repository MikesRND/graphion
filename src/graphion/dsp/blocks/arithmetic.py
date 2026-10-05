"""Arithmetic blocks for the Python/NumPy frame Implementation Profile."""

import numpy as np

from graphion.profile.numpy_frame import Array, Block, C, Control, In, N, Out


class Gain(Block):
    gain: float = Control(default=1.0)
    x: Array = In(C, N)
    y: Array = Out(C, N)

    def output(self):
        np.multiply(self.x, self.gain, out=self.y)
