import numpy as np
import pytest

from examples.ber.ber_tb import main, simulate, sweep
from examples.ber.diagram import BerDiagram
from graphion.dsp.blocks import Gain
from graphion.dsp.communications import bpsk, qpsk
from graphion.profile.numpy_frame import Ctx


def test_diagram_uses_description_without_advancing_model(tmp_path):
    model = simulate(qpsk(), 4, frames=1)
    before = model.prbs.register.copy(), model.counter.total.copy()
    diagram = BerDiagram(headless=True)
    diagram.update(model, "QPSK", 1, 3)
    assert diagram.description == model.describe()
    assert len(diagram.description["connections"]) == 8
    labels = [label.get_text() for label in diagram.axes.texts]
    for block in model.describe()["blocks"][1:]:
        assert block["definition"]["qualname"] in labels
    assert "y: uint8 [1 × 4096]" in labels  # Literal dtype, not the source ctx dtype.
    assert "y: complex128 [1 × 16384]" in labels
    assert "y → ref" in labels
    assert "frame 1 / 3" in diagram.status.get_text()
    diagram.save(tmp_path / "model.png")
    assert (tmp_path / "model.png").read_bytes().startswith(b"\x89PNG")
    np.testing.assert_array_equal(model.prbs.register, before[0])
    np.testing.assert_array_equal(model.counter.total, before[1])


def test_diagram_places_the_optional_viewer(tmp_path):
    plain = BerDiagram(headless=True)
    plain.update(simulate(qpsk(), 4, frames=1), "QPSK", 1, 3)
    assert plain.axes.get_xlim() == pytest.approx((-1.6, 12.1))
    model = simulate(qpsk(), 4, frames=1, viewer_depth=512)
    before = model.viewer.history.copy()
    diagram = BerDiagram(headless=True)
    diagram.update(model, "QPSK", 1, 3)
    assert diagram.description == model.describe()
    assert len(diagram.description["connections"]) == 9
    labels = [label.get_text() for label in diagram.axes.texts]
    for expected in ("viewer", "ConstellationViewer", "depth=512, skip=24", "points: complex128 [1 × 512]"):
        assert expected in labels
    assert labels.count("y → x") == 8  # Every connection but the reference into the counter.
    assert labels.count("y → ref") == 1
    assert diagram.axes.get_xlim() == pytest.approx((-1.6, 15.6))
    diagram.save(tmp_path / "model.png")
    assert (tmp_path / "model.png").read_bytes().startswith(b"\x89PNG")
    np.testing.assert_array_equal(model.viewer.history, before)


def test_diagram_rejects_other_models():
    other = Gain()
    other.setup(Ctx(rate=1, n=8, channels=1))
    with pytest.raises(ValueError, match="configured BerTestbench"):
        BerDiagram(headless=True).update(other, "QPSK", 1, 3)


def test_update_can_leave_event_servicing_to_the_caller(monkeypatch):
    import matplotlib.pyplot as plt

    # Build a window-owning diagram on the test canvas.
    monkeypatch.setattr(plt, "get_backend", lambda: "qtagg")
    diagram = BerDiagram()
    shown = []
    monkeypatch.setattr(plt, "show", lambda **kwargs: shown.append(kwargs))
    model = simulate(qpsk(), 4, frames=1)
    diagram.update(model, "QPSK", 1, 3, service_events=False)
    assert shown == [] and "frame 1 / 3" in diagram.status.get_text()
    diagram.update(model, "QPSK", 2, 3)
    assert shown == [{"block": False}] and "frame 2 / 3" in diagram.status.get_text()
    diagram.close()


def test_diagram_updates_point_and_preserves_closed_window():
    diagram = BerDiagram(headless=True)
    first = simulate(bpsk(), 0, frames=1)
    second = simulate(qpsk(), 4, frames=1, span=8)
    diagram.update(first, "BPSK", 0, 2)
    diagram.update(second, "QPSK", 2, 2)
    assert diagram.description == second.describe()
    assert "QPSK" in diagram.title.get_text() and "4 dB" in diagram.title.get_text()
    assert "Point complete" in diagram.status.get_text()
    labels = [label.get_text() for label in diagram.axes.texts]
    assert "delay=16" in labels
    diagram.finish()
    assert "Sweep complete" in diagram.status.get_text()
    diagram.close()
    diagram.update(first, "BPSK", 1, 2)
    assert diagram.description == second.describe()


def test_progress_observes_complete_frames_and_preserves_results():
    events = []

    def progress(model, name, frame, frames):
        events.append((name, frame, frames))
        if frame:
            assert model.counter.total[0] == max(0, frame * 128 - model.skip)

    expected = sweep([0], error_target=10, frame_size=128)
    assert sweep([0], error_target=10, frame_size=128, progress=progress) == expected
    for name in ("BPSK", "QPSK"):
        point = [event for event in events if event[0] == name]
        assert point[0][1] == 0
        assert point[-1][1] == point[-1][2]


def test_diagram_cli_wiring_and_cleanup(monkeypatch):
    from examples.ber import ber_tb
    from examples.ber import diagram as module

    created = []

    def create():
        view = BerDiagram(headless=True)
        created.append(view)
        return view

    monkeypatch.setattr(module, "BerDiagram", create)
    plotted = []
    monkeypatch.setattr(ber_tb, "plot_results", lambda results, **kwargs: plotted.extend(results))
    assert main(["--diagram", "--stop", "0", "--error-target", "100"]) == 0
    assert len(created) == 1 and len(plotted) == 2
    assert "Sweep complete" in created[0].status.get_text()
    assert created[0]._closed
