import numpy as np
import pytest

from graphion.dsp.filter_design import rrc


@pytest.mark.parametrize("rolloff", [0, 0.25, 0.35, 0.5, 1])
def test_rrc(rolloff):
    taps = rrc(12, 8, rolloff)
    assert taps.dtype == np.float64 and len(taps) == 97
    assert np.isfinite(taps).all()
    np.testing.assert_array_equal(taps, taps[::-1])
    assert np.dot(taps, taps) == pytest.approx(1)
    cascade = np.convolve(taps, taps[::-1])
    assert np.argmax(cascade) == 96
    assert cascade[96] == pytest.approx(1)
    # Independently compare the removable singularity with its analytic limit.
    if rolloff:
        center = 1 + rolloff * (4 / np.pi - 1)
        assert taps[48] / taps[49] == pytest.approx(
            center
            / (
                (np.sin(np.pi / 8 * (1 - rolloff)) + rolloff / 2 * np.cos(np.pi / 8 * (1 + rolloff)))
                / (np.pi / 8 * (1 - (rolloff / 2) ** 2))
            )
        )


@pytest.mark.parametrize("args", [(0, 8, 0.35), (12, 0, 0.35), (12, 8, -0.1), (12, 8, 1.1), (12, 8, np.nan)])
def test_bad_rrc(args):
    with pytest.raises(ValueError):
        rrc(*args)
