import numpy as np
import pytest

from graphion.dsp.blocks import FirInterpolate
from graphion.dsp.filter_design import rrc
from graphion.profile.numpy_frame import Ctx, SetupError
from tests.allocation_helpers import check_storage_regression


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("factor,n,ntaps", [(1, 3, 1), (3, 3, 2), (4, 4, 23), (8, 16, 97)])
def test_interpolator_against_zero_insertion_and_convolution(dtype, factor, n, ntaps):
    rng = np.random.default_rng(32)
    x = rng.normal(size=(2, n * 5)).astype(dtype)
    taps = rng.normal(size=ntaps).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        x += 1j * rng.normal(size=x.shape)
        taps += 1j * rng.normal(size=taps.shape)
    block = FirInterpolate(taps=taps, factor=factor)
    block.setup(Ctx(rate=1000, n=n, channels=2, dtype=dtype))
    frames = []
    for chunk in np.split(x, 5, axis=1):
        block.x = chunk
        frames.append(block.run().copy())
    inserted = np.zeros((2, x.shape[1] * factor), dtype=dtype)
    inserted[:, ::factor] = x
    expected = np.stack([np.convolve(row, taps)[: inserted.shape[1]] for row in inserted])
    bound = ntaps * np.finfo(dtype).eps * np.abs(x).max() * np.abs(taps).sum()
    np.testing.assert_allclose(np.concatenate(frames, axis=1), expected, rtol=0, atol=bound)
    block.reset()
    block.x = x[:, :n]
    np.testing.assert_array_equal(block.run(), frames[0])


@pytest.mark.parametrize("dtype,taps", [(np.int16, [1.0]), (np.float64, [1j])])
def test_interpolator_rejects_unsupported_dtypes(dtype, taps):
    with pytest.raises(SetupError):
        FirInterpolate(taps=np.array(taps), factor=1).setup(Ctx(rate=1, n=8, channels=1, dtype=dtype))


@pytest.mark.parametrize("kwargs", [{"factor": 0}, {"factor": True}, {"factor": 1.5}, {"taps": []}, {"taps": [np.nan]}])
def test_interpolator_validation(kwargs):
    config = {"factor": 2, "taps": [1, 2, 1]} | kwargs
    with pytest.raises(SetupError):
        FirInterpolate(**config).setup(Ctx(rate=1, n=8, channels=1))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_interpolator_storage(dtype):
    block = FirInterpolate(taps=rrc(12, 8, 0.35), factor=8)
    block.setup(Ctx(rate=1, n=65536, channels=2, dtype=dtype))
    block.x = np.ones((2, 65536), dtype=dtype)
    check_storage_regression(block.run, frame_bytes=block.y.nbytes, repeats=3)
