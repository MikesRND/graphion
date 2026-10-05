"""Three independently tuned DDCs sharing one ADC input buffer."""

import numpy as np

from graphion import Array, Block, C, In, N, Out, Param, ParamError

from .ddc import DDC, DECIMATION

DEFAULT_FREQS = (150.3e3, 262.1e3, 405.7e3)


class MultiChannelDDC(Block):
    freqs: tuple = Param(default=DEFAULT_FREQS)
    x: Array = In(C, N)
    y0: Array = Out(C, N / DECIMATION, dtype=np.complex128)
    y1: Array = Out(C, N / DECIMATION, dtype=np.complex128)
    y2: Array = Out(C, N / DECIMATION, dtype=np.complex128)

    @property
    def output_ctx(self):
        return self.ddc0.output_ctx

    def build(self):
        if len(self.freqs) != 3:
            raise ParamError("MultiChannelDDC needs three tuning frequencies")
        self.ddc0 = self.child("ddc0", DDC(freq=self.freqs[0]))
        self.ddc1 = self.child("ddc1", DDC(freq=self.freqs[1]))
        self.ddc2 = self.child("ddc2", DDC(freq=self.freqs[2]))
        for i, branch in enumerate(self.children.values()):
            self.connect(self, "x", branch, "x")
            self.connect(branch, "y", self, f"y{i}")

    def process(self):
        for branch in self.children.values():
            branch.run()
