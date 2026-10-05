import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from examples.ddc.ddc_tb import ADC_RATE
from examples.ddc.ddc_tb import Testbench as DdcTestbench
from examples.ddc.ddc_tb import make_plot as ddc_plot
from examples.plotting import LivePlotter, PlotPanel, PlotSeries
from examples.time_domain import FRAME_LENGTH, RATE, TimeDomain
from examples.time_domain import make_plot as time_plot
from graphion import Ctx


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "plotting requires optional matplotlib")
class PlottingTests(unittest.TestCase):
    def test_step_precedes_reading_and_reuses_artists(self):
        data = np.zeros(3)
        events = []

        def step(frame, elapsed_s):
            events.append(("step", frame, elapsed_s))
            data[:] = frame

        def read():
            events.append(("read",))
            return data

        plotter = LivePlotter([PlotPanel([PlotSeries([0, 1, 2], read)])], step, headless=True)
        line = plotter.lines[0][0]
        plotter.update(7, elapsed_s=1.25)
        self.assertEqual(events, [("step", 7, 1.25), ("read",)])
        np.testing.assert_array_equal(line.get_ydata(), [7, 7, 7])
        plotter.update(8, elapsed_s=1.3)
        self.assertIs(plotter.lines[0][0], line)
        self.assertEqual(len(plotter.axes[0].lines), 1)
        np.testing.assert_array_equal(line.get_ydata(), [8, 8, 8])

    def test_elapsed_time_uses_clock_instead_of_frame_number(self):
        elapsed = []
        plotter = LivePlotter(
            [PlotPanel([PlotSeries([0, 1], lambda: [0, 1])])],
            lambda frame, seconds: elapsed.append(seconds),
            headless=True,
        )
        with patch("examples.plotting.live.perf_counter", side_effect=[100.0, 100.75]):
            plotter.update(0)
            plotter.update(9000)
        self.assertEqual(elapsed, [0, 0.75])

    def test_multiple_series_autoscale_and_fixed_limits(self):
        data = np.array([1.0, 2.0, 3.0])
        panel = PlotPanel(
            [
                PlotSeries([0, 1, 2], lambda: data, label="Positive"),
                PlotSeries([0, 1, 2], lambda: -data, label="Negative"),
            ],
            xlim=(0, 2),
        )
        plotter = LivePlotter([panel], headless=True)
        plotter.refresh()
        data *= 100
        plotter.refresh()
        self.assertEqual(len(plotter.axes[0].lines), 2)
        self.assertEqual(plotter.axes[0].get_xlim(), (0, 2))
        lower, upper = plotter.axes[0].get_ylim()
        self.assertLess(lower, -300)
        self.assertGreater(upper, 300)

    def test_save_does_not_advance_model_or_change_backend(self):
        import matplotlib

        backend = matplotlib.get_backend()
        calls = []
        plotter = LivePlotter(
            [PlotPanel([PlotSeries([0, 1], lambda: [1, 0])])],
            lambda frame, elapsed: calls.append(frame),
            headless=True,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "plot.png"
            plotter.save(path)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(calls, [])
        self.assertEqual(matplotlib.get_backend(), backend)
        with self.assertRaises(ValueError):
            plotter.show()

    def test_live_plot_reports_missing_gui_backend(self):
        with patch("matplotlib.pyplot.get_backend", return_value="Agg"):
            plotter = LivePlotter([PlotPanel([PlotSeries([0, 1], lambda: [1, 0])])])
            with self.assertRaisesRegex(RuntimeError, "GUI Matplotlib backend"):
                plotter.show()
            plotter.close()

    def test_ddc_layout_and_annotation_follow_elapsed_time(self):
        tb = DdcTestbench(noise=0)
        tb.setup(Ctx(rate=ADC_RATE, n=4096, channels=1))
        tb.scope0.avg = 0
        for _ in range(4):
            tb.run()
        plotter = ddc_plot(tb, headless=True)
        self.assertEqual(len(plotter.axes), 4)
        self.assertEqual(plotter.axes[0].get_subplotspec().colspan.stop, 3)
        sample = tb.stim.sample
        plotter.update(9999, elapsed_s=7.5)
        self.assertEqual(tb.stim.sample, sample + 4096)
        self.assertAlmostEqual(tb.dut.ddc0.tune.freq, tb.freqs[0] + 25_000)
        self.assertIn("175.3 kHz", plotter.axes[0].texts[0].get_text())
        peak = plotter.lines[1][0].get_xdata()[np.argmax(plotter.lines[1][0].get_ydata())]
        self.assertAlmostEqual(peak, -20, delta=0.25)

    def test_second_example_uses_overlay_panels(self):
        model = TimeDomain()
        model.setup(Ctx(rate=RATE, n=FRAME_LENGTH, channels=1))
        model.run()
        plotter = time_plot(model, headless=True)
        plotter.update(0, elapsed_s=2.5)
        self.assertEqual(len(plotter.axes), 2)
        self.assertEqual([len(lines) for lines in plotter.lines], [2, 2])
        np.testing.assert_array_equal(plotter.lines[0][0].get_ydata(), model.tone.y[0])
        np.testing.assert_array_equal(plotter.lines[0][1].get_ydata(), model.filter.y[0])


if __name__ == "__main__":
    unittest.main()
