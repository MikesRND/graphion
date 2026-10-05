"""Compare a sine wave and its OnePoleLowpass output in time and frequency."""

from fractions import Fraction

import numpy as np

from examples.blocks import Spectrum, Tone
from examples.plotting import LivePlotter, PlotPanel, PlotSeries
from graphion import Block, Ctx
from graphion.dsp.blocks import OnePoleLowpass

RATE = Fraction(1024)
FRAME_LENGTH = 256


class TimeDomain(Block):
    def build(self):
        self.tone = self.child("tone", Tone(freq=10))
        self.filter = self.child("filter", OnePoleLowpass(pole=0.95))
        self.scope_in = self.child("scope_in", Spectrum(avg=0.7))
        self.scope_out = self.child("scope_out", Spectrum(avg=0.7))
        self.connect(self.tone, "y", self.filter, "x")
        self.connect(self.tone, "y", self.scope_in, "x")
        self.connect(self.filter, "y", self.scope_out, "x")

    def process(self):
        self.tone.run()
        self.filter.run()
        self.scope_in.run()
        self.scope_out.run()


def make_plot(model: TimeDomain, *, headless: bool = False) -> LivePlotter:
    time_ms = np.arange(model.ctx.n) * 1000 / float(model.ctx.rate)
    frequencies = model.scope_in.frequencies
    panels = [
        PlotPanel(
            series=[
                PlotSeries(time_ms, lambda: model.tone.y[0], label="Input"),
                PlotSeries(time_ms, lambda: model.filter.y[0], label="Filtered"),
            ],
            title="Time-domain waveforms",
            xlabel="Time within frame (ms)",
            ylabel="Amplitude",
            xlim=(0, float(time_ms[-1])),
            ylim=(-1.1, 1.1),
        ),
        PlotPanel(
            series=[
                PlotSeries(frequencies, lambda: model.scope_in.psd[0], label="Input"),
                PlotSeries(frequencies, lambda: model.scope_out.psd[0], label="Filtered"),
            ],
            title="Two-sided spectra",
            xlabel="Frequency (Hz)",
            ylabel="PSD (dB/Hz)",
            xlim=(-40, 40),
            ylim=(-100, 0),
        ),
    ]

    def advance(frame: int, elapsed_s: float) -> None:
        model.tone.amplitude = 0.7 + 0.3 * np.sin(2 * np.pi * elapsed_s / 10)
        model.run()

    return LivePlotter(panels, advance, headless=headless, title="Sine wave → OnePoleLowpass · 1.024 kHz")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="PNG", help="save a static plot without opening a window")
    args = parser.parse_args()
    model = TimeDomain()
    model.setup(Ctx(rate=RATE, n=FRAME_LENGTH, channels=1))
    for _ in range(40 if args.save else 1):
        model.run()
    plotter = make_plot(model, headless=bool(args.save))
    if args.save:
        plotter.save(args.save)
    else:
        plotter.show()
        plotter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
