import numpy as np
import pytest

from graphion.dsp.blocks import Prbs
from graphion.profile.numpy_frame import SetupError
from tests.allocation_helpers import check_storage_regression
from tests.dsp_helpers import context


def pn(order, n, seed=None, channels=1, invert=False):
    block = Prbs(order=order, seed=seed, invert=invert)
    block.setup(context(n, channels))
    return block


def test_prbs_prefix():
    assert "".join(str(v) for v in pn(15, 30).run()[0]) == "111111111111111000000000000001"


@pytest.mark.parametrize("order", [7, 9, 15])
def test_prbs_maximal_period(order):
    period = 2**order - 1
    block = pn(order, period)
    first = block.run().copy()
    np.testing.assert_array_equal(block.run(), first)
    windows = np.lib.stride_tricks.sliding_window_view(np.concatenate([first[0], first[0, : order - 1]]), order)
    assert len(np.unique(windows, axis=0)) == period
    assert first.sum() == 2 ** (order - 1)


@pytest.mark.parametrize("order", [7, 9, 15, 23, 31])
@pytest.mark.parametrize("n", [1, 5, 64, 4096])
def test_prbs_continuity_and_replay(order, n):
    full = pn(order, n * 4, seed=[1, 19], channels=2)
    chunked = pn(order, n, seed=[1, 19], channels=2)
    expected = full.run().copy()
    actual = np.concatenate([chunked.run().copy() for _ in range(4)], axis=1)
    np.testing.assert_array_equal(actual, expected)
    # Independent scalar register recurrence establishes orientation for arbitrary seeds.
    q = {7: 6, 9: 5, 15: 14, 23: 18, 31: 28}[order]
    reg = 19
    reference = []
    for _ in range(n * 4):
        reference.append(reg >> (order - 1))
        reg = ((reg << 1) | (((reg >> (order - 1)) ^ (reg >> (q - 1))) & 1)) & ((1 << order) - 1)
    np.testing.assert_array_equal(actual[1], reference)
    chunked.reset()
    np.testing.assert_array_equal(chunked.run(), expected[:, :n])
    np.testing.assert_array_equal(pn(order, n * 4, seed=[1, 19], channels=2, invert=True).run(), expected ^ 1)
    broadcast = pn(order, n, seed=19, channels=2).run()
    np.testing.assert_array_equal(broadcast[0], broadcast[1])


@pytest.mark.parametrize(
    "kwargs",
    [{"seed": 0}, {"seed": -1}, {"seed": 32768}, {"seed": [1, 2]}, {"seed": 1.5}, {"order": 8}, {"order": 15.0}],
)
def test_prbs_invalid(kwargs):
    with pytest.raises(SetupError):
        Prbs(**kwargs).setup(context())


def test_prbs_storage():
    block = Prbs()
    block.setup(context(65536, 2, np.complex128))
    check_storage_regression(block.run, frame_bytes=65536, repeats=2)
