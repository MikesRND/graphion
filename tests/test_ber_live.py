import math

import numpy as np
import pytest

from examples.ber.live import LiveRun, main, make_models, make_plot
from graphion.dsp.communications import esn0_to_snr_db
from tests.animation_helpers import run_to_completion

SMALL = {"depth": 32, "frame_size": 128}


def advanced(ticks, esn0_db=10.0):
    """Reference models run directly, without any plotting."""
    models = make_models(esn0_db, **SMALL)
    for _ in range(ticks):
        for model in models.values():
            model.run()
    return models


def assert_same_state(models, reference, ticks):
    assert list(models) == list(reference) == ["BPSK", "QPSK"]
    for name, model in models.items():
        expected = reference[name]
        assert model.counter.total[0] == expected.counter.total[0] == max(0, ticks * 128 - model.skip)
        np.testing.assert_array_equal(model.counter.errors, expected.counter.errors)
        np.testing.assert_array_equal(model.prbs.register, expected.prbs.register)
        np.testing.assert_array_equal(model.viewer.points, expected.viewer.points)


def test_equal_esn0_and_constant_noise_settings():
    models = make_models(7.0, **SMALL)
    assert models["BPSK"].ebn0_db == pytest.approx(7.0)
    assert models["BPSK"].ebn0_db - models["QPSK"].ebn0_db == pytest.approx(10 * math.log10(2))
    settings = {}
    for name, model in models.items():
        assert model.viewer.depth == 32
        assert model.chan.snr_db == pytest.approx(float(esn0_to_snr_db(7.0, model.sps)))
        settings[name] = (model.chan.snr_db, model.chan.signal_power, model.ebn0_db)
    assert models["BPSK"].chan.signal_power == pytest.approx(models["QPSK"].chan.signal_power)
    run = LiveRun(models)
    for tick in range(1, 6):
        run.advance()
        assert run.ticks == tick
        for name, model in models.items():
            assert (model.chan.snr_db, model.chan.signal_power, model.ebn0_db) == settings[name]


def test_display_starts_empty_and_reading_never_advances(tmp_path):
    run = LiveRun(make_models(**SMALL))
    plotter = make_plot(run, 10.0, ticks=2, headless=True)
    fresh = advanced(0)
    plotter.refresh()
    plotter.save(tmp_path / "before.png")
    assert run.ticks == 0
    assert "tick 0 / 2" in plotter.figure.get_suptitle()
    for (received, ideal), (name, model) in zip(plotter.lines, run.models.items()):
        assert np.isnan(received.get_xdata()).all()  # Nothing is captured until the first tick.
        np.testing.assert_array_equal(ideal.get_xdata(), np.real(model.constellation))
        np.testing.assert_array_equal(model.prbs.register, fresh[name].prbs.register)
    assert "BER —" in plotter.axes[0].get_title()
    plotter.update(0, elapsed_s=0.0)
    plotter.refresh()
    plotter.save(tmp_path / "after.png")
    assert run.ticks == 1
    assert "tick 1 / 2" in plotter.figure.get_suptitle()
    assert_same_state(run.models, advanced(1), 1)
    for (received, _), ax, (name, model) in zip(plotter.lines, plotter.axes, run.models.items()):
        np.testing.assert_array_equal(received.get_xdata(), model.viewer.points[0].real)
        np.testing.assert_array_equal(received.get_ydata(), model.viewer.points[0].imag)
        assert ax.get_title().startswith(f"{name} · Eb/N0 {model.ebn0_db:.2f} dB")
    assert (tmp_path / "after.png").read_bytes().startswith(b"\x89PNG")
    for model in run.models.values():
        model.reset()
    plotter.refresh()
    for received, _ in plotter.lines:
        assert np.isnan(received.get_xdata()).all()  # A reset model shows no stale points.


@pytest.mark.parametrize("ticks", [1, 5])
def test_headless_cli_advances_exactly_ticks(ticks, tmp_path, monkeypatch, capsys):
    import matplotlib

    from examples.ber import live

    created = []

    def recording_models(*args, **kwargs):
        created.append(make_models(*args, **kwargs))
        return created[-1]

    monkeypatch.setattr(live, "make_models", recording_models)
    backend = matplotlib.get_backend()
    png = tmp_path / "live.png"
    arguments = ["--ticks", str(ticks), "--frame-size", "128", "--depth", "32", "--save", str(png)]
    assert main(arguments) == 0
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert matplotlib.get_backend() == backend
    assert f"Completed {ticks} of {ticks} ticks" in capsys.readouterr().out
    assert_same_state(created[0], advanced(ticks), ticks)


@pytest.mark.parametrize("ticks", [1, 5])
def test_animation_stops_in_the_same_state_as_the_headless_run(ticks):
    run = LiveRun(make_models(**SMALL))
    plotter = make_plot(run, 10.0, ticks=ticks, headless=True)
    animation = plotter.animate()
    plotter.figure.canvas.draw()
    assert run.ticks == 0  # Starting the animation reads the display; it does not advance.
    assert run_to_completion(animation, plotter.figure, limit=50) == ticks + 1
    assert run.ticks == ticks and animation.event_source is None
    assert_same_state(run.models, advanced(ticks), ticks)
    plotter.close()


def test_closing_the_window_early_reports_completed_ticks(monkeypatch, capsys):
    from examples.plotting import LivePlotter

    def closed_after_two_ticks(plotter):
        plotter.update(0, elapsed_s=0.0)
        plotter.update(1, elapsed_s=0.05)

    monkeypatch.setattr(LivePlotter, "show", closed_after_two_ticks)
    assert main(["--ticks", "9", "--frame-size", "128", "--depth", "32"]) == 0
    assert "Completed 2 of 9 ticks" in capsys.readouterr().out


def test_on_tick_follows_every_completed_tick():
    run = LiveRun(make_models(**SMALL))
    seen = []
    plotter = make_plot(run, 10.0, ticks=3, headless=True, on_tick=lambda: seen.append(run.ticks))
    assert seen == []
    for frame in range(3):
        plotter.update(frame, elapsed_s=0.0)
    plotter.refresh()
    assert seen == [1, 2, 3]


@pytest.mark.parametrize(
    "option,name", [(["--diagram"], "QPSK"), (["--diagram", "bpsk"], "BPSK"), (["--diagram", "QPSK"], "QPSK")]
)
def test_diagram_option_shows_one_model_and_closes_with_the_plot(option, name, monkeypatch, capsys):
    from matplotlib.backend_bases import CloseEvent

    from examples.ber import diagram as module
    from examples.ber.diagram import BerDiagram
    from examples.plotting import LivePlotter

    created = []

    def create():
        created.append(BerDiagram(headless=True))
        return created[-1]

    observed = {}

    def show(plotter):
        view = created[0]
        observed["at start"] = view.status.get_text()
        plotter.update(0, elapsed_s=0.0)
        plotter.update(1, elapsed_s=0.05)
        observed["at end"] = view.status.get_text()
        observed["title"] = view.title.get_text()
        observed["labels"] = [label.get_text() for label in view.axes.texts]
        observed["open during run"] = not view._closed
        plotter.figure.canvas.callbacks.process("close_event", CloseEvent("close_event", plotter.figure.canvas))
        observed["closed with the plot"] = view._closed

    monkeypatch.setattr(module, "BerDiagram", create)
    monkeypatch.setattr(LivePlotter, "show", show)
    assert main(["--ticks", "2", "--frame-size", "128", "--depth", "32", *option]) == 0
    assert len(created) == 1 and created[0]._closed
    assert "frame 0 / 2" in observed["at start"] and "frame 2 / 2" in observed["at end"]
    assert name in observed["title"]
    assert "ConstellationViewer" in observed["labels"] and "depth=32, skip=24" in observed["labels"]
    assert observed["open during run"] and observed["closed with the plot"]
    assert "Completed 2 of 2 ticks" in capsys.readouterr().out


def test_diagram_is_closed_when_the_plot_cannot_open(monkeypatch):
    from examples.ber import diagram as module
    from examples.ber.diagram import BerDiagram
    from examples.plotting import LivePlotter

    created = []

    def create():
        created.append(BerDiagram(headless=True))
        return created[-1]

    def no_display(plotter):
        raise RuntimeError("no display")

    monkeypatch.setattr(module, "BerDiagram", create)
    monkeypatch.setattr(LivePlotter, "show", no_display)
    with pytest.raises(RuntimeError, match="no display"):
        main(["--ticks", "2", "--frame-size", "128", "--depth", "32", "--diagram"])
    assert created[0]._closed


@pytest.mark.parametrize(
    "args",
    [
        ["--diagram", "ask"],
        ["--diagram", "--save", "live.png"],
        ["--ticks", "0"],
        ["--ticks", "-3"],
        ["--ticks", "1.5"],
        ["--depth", "0"],
        ["--depth", "-1"],
        ["--frame-size", "0"],
        ["--frame-size", "129"],
        ["--esn0-db", "nan"],
        ["--esn0-db", "inf"],
        ["--seed", "-1"],
    ],
)
def test_invalid_cli_arguments_are_usage_errors(args, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "error:" in error and "Traceback" not in error
