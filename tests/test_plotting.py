import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from examples.ddc.ddc_tb import ADC_RATE
from examples.ddc.ddc_tb import Testbench as DdcTestbench
from examples.ddc.ddc_tb import make_plot as ddc_plot
from examples.plotting import LivePlotter, PlotPanel, PlotSeries, constellation_panel
from examples.time_domain import FRAME_LENGTH, RATE, TimeDomain
from examples.time_domain import make_plot as time_plot
from graphion import Ctx
from graphion.dsp.blocks import ConstellationViewer
from tests.animation_helpers import run_to_completion
from tests.typing_helpers import misuse


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

    def test_marker_series_reads_both_getters_and_panel_fixes_aspect(self):
        points = np.array([1 + 2j, -3 + 4j])
        panel = PlotPanel(
            [PlotSeries(lambda: points.real, lambda: points.imag, marker=".")],
            xlim=(-5, 5),
            ylim=(-5, 5),
            aspect="equal",
        )
        plotter = LivePlotter([panel], headless=True)
        line = plotter.lines[0][0]
        plotter.refresh()
        np.testing.assert_array_equal(line.get_xdata(), [1, -3])
        np.testing.assert_array_equal(line.get_ydata(), [2, 4])
        self.assertEqual((line.get_marker(), line.get_linestyle()), (".", "None"))
        self.assertEqual(plotter.axes[0].get_aspect(), 1.0)
        points *= 2
        plotter.refresh()
        np.testing.assert_array_equal(line.get_xdata(), [2, -6])
        plain = LivePlotter([PlotPanel([PlotSeries([0, 1], lambda: [1, 0])])], headless=True)
        self.assertEqual((plain.lines[0][0].get_marker(), plain.lines[0][0].get_linestyle()), ("None", "-"))
        self.assertEqual(plain.axes[0].get_aspect(), "auto")

    def test_constellation_panel_follows_viewer_history(self):
        viewer = ConstellationViewer(depth=4)
        viewer.setup(Ctx(rate=1, n=4, channels=2, dtype=np.complex128))
        viewer.x = np.array([[1, 1j, -1, -1j], [2, 2j, -2, -2j]])
        reference = [1 + 1j, -1 - 1j]
        plotter = LivePlotter([constellation_panel(viewer, channel=1, reference=reference, title="IQ")], headless=True)
        received, ideal = plotter.lines[0]
        plotter.refresh()
        self.assertTrue(np.isnan(received.get_xdata()).all())  # Nothing is drawn before the first run.
        np.testing.assert_array_equal(ideal.get_xdata(), [1, -1])
        np.testing.assert_array_equal(ideal.get_ydata(), [1, -1])
        viewer.run()
        plotter.refresh()
        np.testing.assert_array_equal(received.get_xdata(), [2, 0, -2, 0])
        np.testing.assert_array_equal(received.get_ydata(), [0, 2, 0, -2])
        viewer.reset()
        plotter.refresh()
        self.assertTrue(np.isnan(received.get_xdata()).all())  # A reset empties the display.
        ax = plotter.axes[0]
        limit = 1.6 * abs(1 + 1j)
        self.assertEqual(ax.get_xlim(), (-limit, limit))
        self.assertEqual(ax.get_ylim(), (-limit, limit))
        self.assertEqual((ax.get_aspect(), ax.get_title()), (1.0, "IQ"))
        fixed = constellation_panel(viewer, limit=3)
        self.assertEqual((fixed.xlim, fixed.ylim, len(fixed.series)), ((-3, 3), (-3, 3), 1))
        self.assertIsNone(constellation_panel(viewer).xlim)  # No reference: autoscale.
        self.assertIsNone(constellation_panel(viewer, reference=[0, 0]).xlim)  # Nothing to scale to.

    def test_constellation_panel_validates_its_arguments(self):
        viewer = ConstellationViewer(depth=4)
        with self.assertRaisesRegex(ValueError, "set up"):
            constellation_panel(viewer)
        viewer.setup(Ctx(rate=1, n=4, channels=2, dtype=np.complex128))
        self.assertEqual(len(constellation_panel(viewer, channel=np.int64(1)).series), 1)
        for channel in (-1, 2, 1.0, True):
            with self.subTest(channel=channel), self.assertRaisesRegex(ValueError, r"channel must be .* \[0, 2\)"):
                constellation_panel(viewer, channel=misuse(channel))
        for limit in (0, -1.5, np.nan, np.inf):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "limit"):
                constellation_panel(viewer, limit=limit)

    def _counting_plotter(self, frames):
        calls = []
        plotter = LivePlotter(
            [PlotPanel([PlotSeries([0, 1], lambda: [len(calls), 0])])],
            lambda frame, elapsed: calls.append(frame),
            headless=True,
            frames=frames,
        )
        return plotter, calls

    def test_finite_animation_arguments(self):
        for frames, repeat in ((7, False), (None, True)):
            plotter, _ = self._counting_plotter(frames)
            with patch("matplotlib.animation.FuncAnimation") as animation:
                self.assertIs(plotter.animate(), animation.return_value)
            self.assertIs(animation.call_args.args[1].__self__, plotter)
            self.assertEqual(animation.call_args.kwargs["frames"], frames)
            self.assertIs(animation.call_args.kwargs["repeat"], repeat)
            self.assertEqual(animation.call_args.kwargs["init_func"], plotter.refresh)
        for frames in (0, -1, 2.0, True):
            with self.assertRaises(ValueError):
                self._counting_plotter(frames)
        plotter, _ = self._counting_plotter(np.int64(3))
        self.assertEqual((plotter.frames, type(plotter.frames)), (3, int))

    def test_finite_animation_frames_and_initialization(self):
        # Saving iterates frames directly, so this checks the count and that
        # initialization does not step. It cannot detect repeated playback.
        from matplotlib.animation import PillowWriter

        plotter, calls = self._counting_plotter(4)
        with TemporaryDirectory() as directory:
            plotter.animate().save(Path(directory) / "frames.gif", writer=PillowWriter(fps=20))
        self.assertEqual(calls, [0, 1, 2, 3])

    def test_finite_animation_stops_after_its_frames(self):
        # The timer path owns stop-or-repeat. A repeating animation never
        # releases its callback, so it would reach the firing limit instead.
        plotter, calls = self._counting_plotter(4)
        animation = plotter.animate()
        plotter.figure.canvas.draw()
        self.assertEqual(calls, [])  # Starting initializes by reading, not stepping.
        firings = run_to_completion(animation, plotter.figure, limit=50)
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual(firings, 5)  # Four frames, then the firing that finds none left.
        self.assertIsNone(animation.event_source)
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
