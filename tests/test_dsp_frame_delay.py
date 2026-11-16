import numpy as np

from graphion.dsp.blocks import FrameDelay
from tests.dsp_helpers import context


def test_frame_delay_call_semantics_and_reset():
    block = FrameDelay()
    block.setup(context(4, 2, np.complex64))
    block.x = np.full((2, 4), 1 + 2j, dtype=np.complex64)
    np.testing.assert_array_equal(block.run(), 0)
    block.x = np.full((2, 4), 3 + 4j, dtype=np.complex64)
    np.testing.assert_array_equal(block.run(), 1 + 2j)
    held = block.held
    block.reset()
    assert held is block.held
    np.testing.assert_array_equal(block.run(), 0)
