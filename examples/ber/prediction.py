"""Finite-filter conditional BER for this example's BPSK/QPSK PRBS15 stream."""

import math

import numpy as np

from graphion.dsp.blocks import Prbs
from graphion.dsp.communications import bpsk, qpsk
from graphion.profile.numpy_frame import Ctx


class FiniteFilterPrediction:
    """Steady-state decision values and BER over a complete symbol period.

    Only the factory BPSK/QPSK mappings are supported: their decisions separate
    into real-axis signs. Noise is circular complex AWGN, as in BerTestbench.
    ``decisions[j]`` corresponds to transmitted symbol j, after delay alignment.
    This model does not execute the simulation's FIR or demapper blocks.
    """

    def __init__(self, constellation, tx_taps, sps):
        table = np.asarray(constellation)
        if np.array_equal(table, bpsk()):
            self.k = 1
        elif np.array_equal(table, qpsk()):
            self.k = 2
        else:
            raise ValueError("prediction supports only the factory BPSK and QPSK mappings")
        if isinstance(sps, bool) or not isinstance(sps, (int, np.integer)) or sps < 1:
            raise ValueError("sps must be a positive integer")
        taps = np.asarray(tx_taps, dtype=np.complex128)
        if taps.ndim != 1 or not taps.size or not np.isfinite(taps).all():
            raise ValueError("tx_taps must be a nonempty finite vector")
        energy = float(np.vdot(taps, taps).real)
        if not math.isfinite(energy) or energy <= 0:
            raise ValueError("tx_taps must have positive finite energy")
        rx_taps = taps.conj()[::-1]
        self.signal_power = float(np.mean(np.abs(table) ** 2)) * energy / sps
        self.rx_energy = float(np.vdot(rx_taps, rx_taps).real)
        delay = (len(taps) + len(rx_taps) - 2) // 2
        self.phase = delay % sps
        self.symbol_delay = (delay - self.phase) // sps
        cascade = np.convolve(taps, rx_taps)[self.phase :: sps]

        # PRBS15's odd period is coprime to k=1 or 2, so both symbol streams
        # repeat after 32767 symbols (QPSK consumes two whole bit periods).
        self.period = (1 << 15) - 1
        source = Prbs()
        source.setup(Ctx(n=self.k * self.period, channels=1, rate=1))
        bits = source.run()[0].reshape(self.period, self.k)
        indices = bits.astype(np.intp) @ (1 << np.arange(self.k - 1, -1, -1))
        self.symbols = table[indices]
        self.decisions = np.zeros(self.period, dtype=np.complex128)
        for index, coefficient in enumerate(cascade):
            self.decisions += coefficient * np.roll(self.symbols, index - self.symbol_delay)
        axes = (
            self.decisions.real[:, None] if self.k == 1 else np.column_stack((self.decisions.real, self.decisions.imag))
        )
        self._margins = axes * (1.0 - 2.0 * bits)
        self._noiseless_ber = float(np.mean((axes < 0) != bits))

    def axis_noise_variance(self, snr_db):
        """Per-axis variance after the matched filter, from sample SNR."""
        if math.isnan(snr_db) or snr_db == -math.inf:
            raise ValueError("snr_db must be finite or +inf")
        return self.signal_power * self.rx_energy / 2 * 10 ** (-snr_db / 10)

    def ber(self, snr_db):
        """Average conditional sign-error probability, including residual ISI."""
        variance = self.axis_noise_variance(snr_db)
        if variance == 0:
            return self._noiseless_ber
        scale = 1 / math.sqrt(2 * variance)
        return math.fsum(0.5 * math.erfc(float(margin) * scale) for margin in self._margins.flat) / self._margins.size
