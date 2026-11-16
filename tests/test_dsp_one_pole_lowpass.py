import numpy as np
import pytest

from graphion.dsp.blocks import OnePoleLowpass
from tests.dsp_helpers import context


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_one_pole_lowpass_frame_continuity_and_reset(dtype):
    signal = np.arange(16, dtype=np.float64).reshape(2, 8).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        signal += 1j
    expected = np.empty_like(signal)
    history = np.zeros(2, dtype=dtype)
    for i in range(8):
        history = 0.75 * history + 0.25 * signal[:, i]
        expected[:, i] = history
    block = OnePoleLowpass(pole=0.75)
    block.setup(context(4, 2, dtype))
    actual = []
    for frame in (signal[:, :4], signal[:, 4:]):
        block.x = frame
        actual.append(block.run().copy())
    np.testing.assert_allclose(np.concatenate(actual, axis=1), expected, rtol=1e-6)
    block.reset()
    block.x = signal[:, :4]
    np.testing.assert_allclose(block.run(), expected[:, :4], rtol=1e-6)
