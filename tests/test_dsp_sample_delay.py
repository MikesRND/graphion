import numpy as np
import pytest

from graphion.dsp.blocks import SampleDelay
from tests.allocation_helpers import check_storage_regression
from tests.dsp_helpers import context


@pytest.mark.parametrize("delay", [0, 1, 4, 7, 21])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.complex128])
def test_sample_delay(delay, dtype):
    x = np.arange(56).reshape(2, 28).astype(dtype)
    block = SampleDelay(delay=delay)
    block.setup(context(7, 2, dtype))
    frames = []
    for row in np.split(x, 4, axis=1):
        block.x = row
        frames.append(block.run().copy())
    expected = np.pad(x, ((0, 0), (delay, 0)))[:, :28]
    np.testing.assert_array_equal(np.concatenate(frames, axis=1), expected)
    block.reset()
    block.x = x[:, :7]
    np.testing.assert_array_equal(block.run(), expected[:, :7])


def test_sample_delay_storage():
    block = SampleDelay(delay=100)
    block.setup(context(65536, 2, np.uint8))
    block.x = np.zeros((2, 65536), dtype=np.uint8)
    check_storage_regression(block.run, frame_bytes=65536, repeats=2)
