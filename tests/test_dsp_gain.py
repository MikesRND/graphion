import numpy as np
import pytest

from graphion.dsp.blocks import Gain
from tests.dsp_helpers import context


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_gain_scales_each_element_and_follows_its_control(dtype):
    x = np.arange(1, 13).reshape(2, 6).astype(dtype)
    block = Gain()
    block.setup(context(6, 2, dtype))
    block.x = x
    np.testing.assert_array_equal(block.run(), x)  # The default gain is one.
    assert block.y.dtype == dtype
    block.gain = 2.5
    np.testing.assert_array_equal(block.run(), x * dtype(2.5))
    block.gain = -1
    np.testing.assert_array_equal(block.run(), -x)
    block.reset()  # Gain holds no history; reset keeps the control.
    np.testing.assert_array_equal(block.run(), -x)
