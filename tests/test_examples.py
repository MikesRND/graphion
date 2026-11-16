import unittest
from fractions import Fraction

import numpy as np

from examples.blocks import DDC, MultiChannelDDC, Tone
from examples.blocks.ddc import PASSBAND_HZ
from examples.ddc.ddc_tb import ADC_RATE, OFFSETS_HZ
from examples.ddc.ddc_tb import Testbench as DdcTestbench
from examples.time_domain import FRAME_LENGTH, RATE, TimeDomain
from graphion import Ctx, SetupError


class DdcExampleTests(unittest.TestCase):
    def test_interfaces_offsets_and_fanout(self):
        tb = DdcTestbench()
        tb.setup(Ctx(rate=ADC_RATE, n=4096, channels=1))
        for _ in range(4):
            self.assertIsNone(tb.run())
        for output in (tb.dut.y0, tb.dut.y1, tb.dut.y2):
            self.assertEqual(output.shape, (1, 256))
            self.assertEqual(output.dtype, np.dtype(np.complex128))
        self.assertEqual(tb.dut.output_ctx.n, 256)
        self.assertEqual(tb.dut.output_ctx.rate, Fraction(64_000))
        self.assertIsInstance(tb.dut.output_ctx.rate, Fraction)
        self.assertEqual(tb.dut.output_ctx.dtype, np.dtype(np.complex128))
        np.testing.assert_allclose(tb.check.offset_hz, OFFSETS_HZ, rtol=0, atol=250)
        self.assertIs(tb.dut.x, tb.scope_in.x)
        self.assertIs(tb.dut.ddc0.x, tb.dut.ddc1.x)
        self.assertIs(tb.dut.ddc1.x, tb.dut.ddc2.x)
        self.assertIs(tb.dut.y0, tb.dut.ddc0.y)
        for scope in tb.scopes:
            self.assertIs(scope.ctx, tb.dut.output_ctx)

    def test_nonintegral_rational_rate_at_every_stage(self):
        dut = MultiChannelDDC()
        rate = Fraction(1_024_000_000, 1001)
        dut.setup(Ctx(rate=rate, n=4096, channels=1))
        for branch in dut.children.values():
            for ctx, divisor in (
                (branch.tune.ctx, 1),
                (branch.hb1.ctx, 1),
                (branch.hb2.ctx, 2),
                (branch.fir.ctx, 4),
                (branch.output_ctx, 16),
            ):
                self.assertEqual(ctx.rate, rate / divisor)
                self.assertIsInstance(ctx.rate, Fraction)
        self.assertEqual(dut.output_ctx.rate, Fraction(64_000_000, 1001))

    def test_chunking_invariance(self):
        n = 4080
        signal = np.random.default_rng(123).standard_normal((2, 4 * n))
        full = MultiChannelDDC()
        full.setup(Ctx(rate=ADC_RATE, n=4 * n, channels=2))
        full.x = signal
        expected = full.run()

        chunked = MultiChannelDDC()
        chunked.setup(Ctx(rate=ADC_RATE, n=n, channels=2))
        frames = []
        for start in range(0, signal.shape[1], n):
            chunked.x = signal[:, start : start + n]
            frames.append(tuple(output.copy() for output in chunked.run()))
        for i, output in enumerate(expected):
            np.testing.assert_allclose(
                np.concatenate([frame[i] for frame in frames], axis=1), output, rtol=1e-9, atol=1e-10
            )

    def test_ddc_band_edge_relative_to_dc(self):
        amplitudes = []
        for offset in (0, -PASSBAND_HZ, PASSBAND_HZ):
            ddc = DDC(freq=150_300)
            ddc.setup(Ctx(rate=ADC_RATE, n=4096, channels=1))
            for frame in range(2):
                t = (frame * 4096 + np.arange(4096)) / float(ADC_RATE)
                ddc.x = np.cos(2 * np.pi * (ddc.freq + offset) * t)[None, :]
                ddc.run()
            reference = np.exp(
                -2j * np.pi * offset * np.arange(ddc.port_spec("y").shape[1]) / float(ddc.output_ctx.rate)
            )
            amplitudes.append(abs(np.mean(ddc.y[0] * reference)))
        for amplitude in amplitudes[1:]:
            self.assertAlmostEqual(float(20 * np.log10(amplitude / amplitudes[0])), -3, delta=0.5)

    def test_interferer_rejection_without_noise(self):
        tb = DdcTestbench(noise=0)
        tb.setup(Ctx(rate=ADC_RATE, n=4096, channels=1))
        tb.run()  # Discard startup, including the spectrum accumulator.
        tb.scope1.reset()
        for _ in range(4):
            tb.run()
        bins = tb.scope1.frequencies
        wanted = np.argmin(abs(bins + 8_000))
        alias = np.argmin(abs(bins + 19_000))
        self.assertLessEqual(tb.scope1.psd[0, alias] - tb.scope1.psd[0, wanted], -60)

    def test_tuning_control_moves_only_first_peak(self):
        tb = DdcTestbench(noise=0)
        tb.setup(Ctx(rate=ADC_RATE, n=4096, channels=1))
        for scope in tb.scopes:
            scope.avg = 0
        tb.run()
        tb.dut.ddc0.tune.freq += 4_000
        tb.run()
        tb.run()  # The filter history now contains only the new tuning.
        np.testing.assert_allclose(tb.check.offset_hz, [1_000, -8_000, 12_000], rtol=0, atol=250)

    def test_reset_restores_noisy_stimulus_and_dsp_state(self):
        tb = DdcTestbench()
        tb.setup(Ctx(rate=ADC_RATE, n=4096, channels=1))
        tb.run()
        first = [scope.psd.copy() for scope in (tb.scope_in, *tb.scopes)]
        tb.run()
        tb.reset()
        tb.run()
        for scope, expected in zip((tb.scope_in, *tb.scopes), first):
            np.testing.assert_array_equal(scope.psd, expected)

    def test_nondivisible_frame_is_rejected(self):
        with self.assertRaises(SetupError):
            MultiChannelDDC().setup(Ctx(rate=ADC_RATE, n=4081, channels=1))


class TimeDomainExampleTests(unittest.TestCase):
    def test_shared_tone_is_continuous_across_frames(self):
        full = Tone(freq=10)
        full.setup(Ctx(rate=RATE, n=4 * FRAME_LENGTH, channels=1))
        expected = full.run()
        chunked = Tone(freq=10)
        chunked.setup(Ctx(rate=RATE, n=FRAME_LENGTH, channels=1))
        actual = np.concatenate([chunked.run().copy() for _ in range(4)], axis=1)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_time_domain_pipeline_and_spectra(self):
        model = TimeDomain()
        model.setup(Ctx(rate=RATE, n=FRAME_LENGTH, channels=1))
        for _ in range(4):
            model.run()
        self.assertEqual(model.filter.y.shape, (1, FRAME_LENGTH))
        self.assertLess(np.mean(model.filter.y**2), np.mean(model.tone.y**2))
        self.assertIs(model.filter.x, model.scope_in.x)
        for scope in (model.scope_in, model.scope_out):
            self.assertEqual(scope.psd.dtype, np.dtype(np.float64))
            self.assertTrue(np.isfinite(scope.psd).all())


if __name__ == "__main__":
    unittest.main()
