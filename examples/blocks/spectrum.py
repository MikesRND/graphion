"""An exponentially averaged, two-sided power spectral density monitor."""

import numpy as np

from graphion import Array, Block, C, Control, Ctx, In, N, Out, SetupError, State


class Spectrum(Block):
    avg: float = Control(default=0.9)
    x: Array = In(C, N)
    psd: Array = Out(C, N, dtype=np.float64)
    acc: Array = State(C, N, dtype=np.float64, default=0.0)

    def setup(self, ctx: Ctx):
        if ctx.n < 3 or not 0 <= self.avg < 1:
            raise SetupError("Spectrum needs at least three samples and avg in [0, 1)")
        self._window = np.hanning(ctx.n)
        self._scale = float(ctx.rate) * np.sum(self._window**2)
        self.frequencies = np.fft.fftshift(np.fft.fftfreq(ctx.n, d=1 / float(ctx.rate)))

    def process(self):
        power = abs(np.fft.fftshift(np.fft.fft(self.x * self._window, axis=1), axes=1)) ** 2 / self._scale
        self.acc[...] = self.avg * self.acc + (1 - self.avg) * power
        self.psd[...] = 10 * np.log10(np.maximum(self.acc, 1e-20))
