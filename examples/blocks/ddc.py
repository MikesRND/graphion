"""One real-ADC-to-complex-baseband down-conversion path."""

import numpy as np

from graphion import Array, Block, C, Ctx, In, N, Out, Param

from .fir_decimate import FirDecimate, lowpass
from .tuner import Tuner

DECIMATION = 16
FIR_INPUT_HZ = 256_000  # 1.024 MHz after two divide-by-two stages.
PASSBAND_HZ = 25_600  # -3 dB near 80% of the 64 kHz output's Nyquist frequency.
TRANSITION_HZ = 38_400 - PASSBAND_HZ
HALFBAND_TAPS = lowpass(23, 0.25)
CHANNEL_TAPS = lowpass(101, (PASSBAND_HZ + 0.1 * TRANSITION_HZ) / FIR_INPUT_HZ)


class DDC(Block):
    freq: float = Param()
    x: Array = In(C, N)
    y: Array = Out(C, N / DECIMATION, dtype=np.complex128)

    def build(self):
        self.tune = self.child("tune", Tuner(freq=self.freq))
        self.hb1 = self.child("hb1", FirDecimate(taps=HALFBAND_TAPS, factor=2))
        self.hb2 = self.child("hb2", FirDecimate(taps=HALFBAND_TAPS, factor=2))
        self.fir = self.child("fir", FirDecimate(taps=CHANNEL_TAPS, factor=4))
        self._stages = (self.hb1, self.hb2, self.fir)
        self.connect(self, "x", self.tune, "x")
        previous = self.tune
        for stage in self._stages:
            self.connect(previous, "y", stage, "x")
            previous = stage
        self.connect(self.fir, "y", self, "y")

    def setup(self, ctx: Ctx):
        self.tune.setup(ctx)
        c = ctx.replace(dtype=np.complex128)
        for stage in self._stages:
            stage.setup(c)
            c = c.replace(n=stage.port_spec("y").shape[1], rate=c.rate / stage.factor)
        self.output_ctx = c

    def process(self):
        self.tune.run()
        for stage in self._stages:
            stage.run()
