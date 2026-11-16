"""Sample-SNR additive white Gaussian noise."""

import math
from copy import deepcopy

import numpy as np

from graphion.profile.numpy_frame import Array, Block, C, Control, In, N, Out, Param, SetupError


class Awgn(Block):
    """Add noise with total sample power signal_power / 10**(snr_db/10).

    Setup copies Generator/BitGenerator state into private storage. Blocks given
    the same state produce identical noise; use independently seeded Generators
    or distinct spawned SeedSequences when independent streams are needed.
    Running or resetting the block never advances or rewinds the caller's RNG.
    """

    snr_db: float = Control()
    signal_power: float = Param()
    seed: object = Param(default=0)
    x: Array = In(C, N)
    y: Array = Out(C, N)
    stream_ports = ("x", "y")

    def setup(self, ctx):
        if ctx.dtype not in (np.float32, np.float64, np.complex64, np.complex128):
            raise SetupError("Awgn needs float32/float64/complex64/complex128 samples")
        if not np.isfinite(self.signal_power) or self.signal_power < 0:
            raise SetupError("signal_power must be finite and nonnegative")
        try:
            self._scale()
        except ValueError as exc:
            raise SetupError(str(exc)) from exc
        self._rng = deepcopy(
            np.random.default_rng(
                self.seed  # pyrefly: ignore[bad-argument-type]  # NumPy validates arbitrary seed objects.
            )
        )
        self._initial_rng_state = deepcopy(self._rng.bit_generator.state)
        self._noise = np.empty((ctx.channels, ctx.n), ctx.dtype)
        self._real: Array = self._noise.view(
            np.float32 if ctx.dtype.itemsize // (2 if ctx.dtype.kind == "c" else 1) == 4 else np.float64
        )

    def _scale(self):
        if math.isnan(self.snr_db) or self.snr_db == -math.inf:
            raise ValueError("snr_db must be finite or +inf")
        return math.sqrt(self.signal_power / (2 if self.ctx.dtype.kind == "c" else 1)) * 10 ** (-self.snr_db / 20)

    def reset(self):
        self._rng.bit_generator.state = self._initial_rng_state

    def process(self):
        scale = self._scale()
        if scale == 0:
            np.copyto(self.y, self.x)
            return
        self._rng.standard_normal(out=self._real, dtype=self._real.dtype)
        np.multiply(self._noise, scale, out=self._noise)
        np.add(self.x, self._noise, out=self.y)
