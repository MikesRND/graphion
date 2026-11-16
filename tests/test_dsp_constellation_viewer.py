import numpy as np
import pytest

from graphion.dsp.blocks import ConstellationViewer
from graphion.profile.numpy_frame import SetupError
from tests.allocation_helpers import check_storage_regression
from tests.dsp_helpers import context


def recent(data, depth, skip):
    """The last depth samples after skip leading ones, NaN-padded at the front."""
    kept = data[:, skip:][:, -depth:]
    expected = np.full((data.shape[0], depth), np.nan, dtype=data.dtype)
    expected[:, depth - kept.shape[1] :] = kept
    return expected


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize(
    "depth,skip",
    [
        (3, 0),  # shorter than a frame
        (5, 0),
        (12, 0),  # longer than a frame; NaN until filled
        (4, 2),  # skip ends inside the first frame
        (4, 7),  # skip swallows a frame and ends inside the next
        (6, 5),  # skip swallows exactly one frame
        (12, 11),
        (3, 20),  # everything skipped
    ],
)
def test_constellation_viewer(depth, skip, dtype):
    rng = np.random.default_rng(depth * 100 + skip)
    data = (rng.normal(size=(2, 20)) + 1j * rng.normal(size=(2, 20))).astype(dtype)
    viewer = ConstellationViewer(depth=depth, skip=skip)
    viewer.setup(context(5, 2, dtype))
    assert np.isnan(viewer.history).all()

    def stream():
        for stop in range(5, 25, 5):
            viewer._line.fill(99)  # Stale scratch must never reach the retained points.
            viewer.x = data[:, stop - 5 : stop]
            points = viewer.run()
            assert np.shares_memory(points, viewer.points) and points.dtype == dtype and points.shape == (2, depth)
            np.testing.assert_array_equal(points, recent(data[:, :stop], depth, skip))
            np.testing.assert_array_equal(viewer.history, points)  # A host may read either between runs.

    stream()
    final = viewer.points.copy()
    viewer.reset()
    assert np.isnan(viewer.history).all() and viewer.remaining == skip
    np.testing.assert_array_equal(viewer.points, final)  # Reset leaves the output alone.
    stream()
    np.testing.assert_array_equal(viewer.points, final)


@pytest.mark.parametrize(
    "kwargs,dtype,match",
    [
        ({}, np.float64, "complex64 or complex128"),
        ({}, np.uint8, "complex64 or complex128"),
        ({"depth": 0}, np.complex128, "depth"),
        ({"depth": -4}, np.complex128, "depth"),
        ({"depth": 2.0}, np.complex128, "depth"),
        ({"depth": True}, np.complex128, "depth"),
        ({"skip": -1}, np.complex128, "skip"),
        ({"skip": 1.5}, np.complex128, "skip"),
        ({"skip": False}, np.complex128, "skip"),
    ],
)
def test_constellation_viewer_rejects_invalid_setup(kwargs, dtype, match):
    with pytest.raises(SetupError, match=match):
        ConstellationViewer(**kwargs).setup(context(8, 1, dtype))


def test_constellation_viewer_storage():
    block = ConstellationViewer(depth=100)
    block.setup(context(65536, 2, np.complex128))
    block.x = np.zeros((2, 65536), dtype=np.complex128)
    check_storage_regression(block.run, frame_bytes=65536, repeats=2)
