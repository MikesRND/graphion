"""A runnable DDC testbench with live or headless spectrum plotting."""

from fractions import Fraction

import numpy as np

from examples.blocks import MultiChannelDDC, Spectrum
from examples.blocks.ddc import PASSBAND_HZ
from examples.blocks.multichannel_ddc import DEFAULT_FREQS
from examples.plotting import LivePlotter, PlotPanel, PlotSeries
from graphion import Array, Axis, Block, C, Ctx, In, N, Out, Param, SetupError, State

ADC_RATE = Fraction(1_024_000)
OFFSETS_HZ = np.array([5_000, -8_000, 12_000], dtype=np.float64)


class Stimulus(Block):
    freqs: tuple = Param(default=DEFAULT_FREQS)
    noise: float = Param(default=0.01)
    seed: int = Param(default=0)
    y: Array = Out(C, N)
    sample: int = State(default=0)

    def setup(self, ctx: Ctx):
        if self.ctx.dtype.kind != "f" or not np.isfinite(self.noise) or self.noise < 0:
            raise SetupError("Stimulus needs real floating samples and nonnegative finite noise RMS")
        self._k = np.arange(self.ctx.n)
        self._rng = np.random.default_rng(self.seed)
        self._initial_rng_state = self._rng.bit_generator.state

    def reset(self):
        self._rng.bit_generator.state = self._initial_rng_state

    def process(self):
        t = (self.sample + self._k) / float(self.ctx.rate)
        frequencies = np.asarray(self.freqs) + OFFSETS_HZ
        self.y[...] = np.cos(2 * np.pi * frequencies[:, None] * t).sum(axis=0)
        self.y[...] += 10 * np.cos(2 * np.pi * (self.freqs[1] + 45_000) * t)
        if self.noise:
            self.y[...] += self.noise * self._rng.standard_normal(self.y.shape)
        self.sample += self.ctx.n


class PeakCheck(Block):
    psd0: Array = In(C, Axis("nfft"), dtype=np.float64)
    psd1: Array = In(C, Axis("nfft"), dtype=np.float64)
    psd2: Array = In(C, Axis("nfft"), dtype=np.float64)
    offset_hz: Array = Out(3, dtype=np.float64)

    @property
    def nfft(self):
        return self.ctx.n

    def setup(self, ctx: Ctx):
        self._frequencies = np.fft.fftshift(np.fft.fftfreq(ctx.n, d=1 / float(ctx.rate)))

    def output(self):
        self.offset_hz[...] = [self._frequencies[np.argmax(psd[0])] for psd in (self.psd0, self.psd1, self.psd2)]


class Testbench(Block):
    freqs: tuple = Param(default=DEFAULT_FREQS)
    noise: float = Param(default=0.01)
    seed: int = Param(default=0)

    def build(self):
        self.stim = self.child("stim", Stimulus(freqs=self.freqs, noise=self.noise, seed=self.seed))
        self.dut = self.child("dut", MultiChannelDDC(freqs=self.freqs))
        self.scope_in = self.child("scope_in", Spectrum())
        self.scope0 = self.child("scope0", Spectrum())
        self.scope1 = self.child("scope1", Spectrum())
        self.scope2 = self.child("scope2", Spectrum())
        self.check = self.child("check", PeakCheck())
        self.scopes = (self.scope0, self.scope1, self.scope2)
        self.connect(self.stim, "y", self.dut, "x")
        self.connect(self.stim, "y", self.scope_in, "x")
        for i, scope in enumerate(self.scopes):
            self.connect(self.dut, f"y{i}", scope, "x")
            self.connect(scope, "psd", self.check, f"psd{i}")

    def setup(self, ctx: Ctx):
        self.dut.setup(ctx)
        self.stim.setup(ctx)
        self.scope_in.setup(ctx)
        for scope in self.scopes:
            scope.setup(self.dut.output_ctx)
        self.check.setup(self.dut.output_ctx)

    def process(self):
        self.stim.run()
        self.dut.run()
        self.scope_in.run()
        for scope in self.scopes:
            scope.run()
        self.check.run()


def make_plot(tb: Testbench, *, headless: bool = False) -> LivePlotter:
    """Describe the DDC panels and keep its tuning/annotations in the demo."""
    from matplotlib.transforms import Affine2D

    def spectrum_series(scope: Spectrum, color: str) -> PlotSeries:
        return PlotSeries(scope.frequencies / 1e3, lambda: scope.psd[0], color=color)

    monitors = (tb.scope_in, *tb.scopes)
    panels = [
        PlotPanel(
            series=[spectrum_series(scope, f"C{max(i - 1, 0)}")],
            title="Real ADC input (positive frequencies)" if i == 0 else f"DDC {i - 1} baseband",
            xlabel="Frequency (kHz)",
            ylabel="PSD (dB/Hz)",
            xlim=(
                (0, float(scope.ctx.rate) / 2e3)
                if i == 0
                else (float(scope.frequencies[0] / 1e3), float(scope.frequencies[-1] / 1e3))
            ),
            ylim=(-120, 0),
            position=(0, 0) if i == 0 else (1, i - 1),
            span=(1, 3) if i == 0 else (1, 1),
        )
        for i, scope in enumerate(monitors)
    ]

    def advance(frame: int, elapsed_s: float) -> None:
        frequency = tb.freqs[0] + 25_000 * np.sin(2 * np.pi * elapsed_s / 30)
        tb.dut.ddc0.tune.freq = frequency
        tb.run()
        bands[0].set_transform(
            Affine2D().translate((frequency - tb.freqs[0]) / 1e3, 0) + plotter.axes[0].get_xaxis_transform()
        )
        readout.set_text(f"DDC 0 tuner: {frequency / 1e3:.1f} kHz")

    plotter = LivePlotter(
        panels,
        advance,
        headless=headless,
        title="Three-channel digital down converter · 1.024 MHz → 64 kHz",
    )
    bands = [
        plotter.axes[0].axvspan((freq - PASSBAND_HZ) / 1e3, (freq + PASSBAND_HZ) / 1e3, alpha=0.15, color=f"C{i}")
        for i, freq in enumerate(tb.freqs)
    ]
    readout = plotter.axes[0].text(
        0.02, 0.94, f"DDC 0 tuner: {tb.freqs[0] / 1e3:.1f} kHz", transform=plotter.axes[0].transAxes
    )
    return plotter


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="PNG", help="save a static plot without opening a window")
    args = parser.parse_args()

    tb = Testbench()
    tb.setup(Ctx(rate=ADC_RATE, n=4096, channels=1))
    for scope in (tb.scope_in, *tb.scopes):
        scope.avg = 0.9 if args.save else 0.5
    if not args.save:
        tb.scope0.avg = 0.0
    for _ in range(40 if args.save else 1):
        tb.run()
    outputs = (tb.dut.y0, tb.dut.y1, tb.dut.y2)
    print(f"Output rate: {tb.dut.output_ctx.rate!r} Hz; shapes: {[output.shape for output in outputs]}")

    plotter = make_plot(tb, headless=bool(args.save))

    if args.save:
        passed = bool(
            np.all(
                np.abs(tb.check.offset_hz - OFFSETS_HZ)
                <= float(tb.dut.output_ctx.rate) / tb.dut.port_spec("y0").shape[1]
            )
        )
        print(f"{'PASS' if passed else 'FAIL'}: peak offsets {tb.check.offset_hz} Hz; expected {OFFSETS_HZ} Hz")
        plotter.save(args.save)
        return 0 if passed else 1

    plotter.show()
    plotter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
