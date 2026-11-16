import unittest

import numpy as np
import pytest

from graphion.dsp.blocks import FirDecimate
from graphion.dsp.filter_design import rrc
from graphion.profile.numpy_frame import Ctx, SetupError
from tests.allocation_helpers import check_storage_regression


class FirDecimateTests(unittest.TestCase):
    FRAMES = 5

    @staticmethod
    def signal(rng, channels, length, dtype, offset):
        """Noise on a common offset; a large offset must cancel in zero-sum taps."""
        values = offset + rng.standard_normal((channels, length))
        if dtype is np.complex128:
            values = values + 1j * (offset + rng.standard_normal((channels, length)))
        return values.astype(dtype)

    def streamed(self, block, signal, n):
        frames = []
        for start in range(0, signal.shape[1], n):
            block.x = signal[:, start : start + n]
            frames.append(block.run().copy())
        return np.concatenate(frames, axis=1)

    def test_streaming_matches_causal_convolution(self):
        rng = np.random.default_rng(7)
        for ntaps, factor, n in ((1, 1, 8), (1, 4, 8), (5, 2, 64), (23, 2, 4), (23, 4, 8), (101, 4, 256)):
            for channels in (1, 3):
                for dtype in (np.float64, np.complex128):
                    for offset in (0.0, 1e6):
                        with self.subTest(
                            ntaps=ntaps, factor=factor, n=n, channels=channels, dtype=dtype.__name__, offset=offset
                        ):
                            taps = rng.standard_normal(ntaps)
                            if offset and ntaps > 1:
                                taps -= taps.mean()
                            signal = self.signal(rng, channels, self.FRAMES * n, dtype, offset)
                            block = FirDecimate(taps=taps, factor=factor)
                            block.setup(Ctx(rate=1000, n=n, channels=channels, dtype=dtype))
                            self.assertEqual(block.history.shape, (channels, ntaps - 1))
                            actual = self.streamed(block, signal, n)
                            expected = np.stack([np.convolve(row, taps)[: signal.shape[1] : factor] for row in signal])
                            self.assertEqual(actual.dtype, np.dtype(dtype))
                            # Rounding scales with the operands, not with an output that may cancel.
                            bound = ntaps * np.finfo(np.float64).eps * np.abs(signal).max() * np.abs(taps).sum()
                            np.testing.assert_allclose(actual, expected, rtol=0, atol=bound)

    def test_reset_replays_exactly(self):
        rng = np.random.default_rng(8)
        for dtype in (np.float64, np.complex128):
            with self.subTest(dtype=dtype.__name__):
                block = FirDecimate(taps=rng.standard_normal(23), factor=2)
                block.setup(Ctx(rate=1000, n=64, channels=2, dtype=dtype))
                signal = self.signal(rng, 2, 4 * 64, dtype, 0.0)
                first = self.streamed(block, signal, 64)
                block.reset()
                np.testing.assert_array_equal(self.streamed(block, signal, 64), first)

    def test_processing_avoids_frame_sized_temporaries(self):
        rng = np.random.default_rng(9)
        for dtype in (np.float64, np.complex128):
            with self.subTest(dtype=dtype.__name__):
                block = FirDecimate(taps=rng.standard_normal(23), factor=2)
                block.setup(Ctx(rate=1000, n=65536, channels=2, dtype=dtype))
                block.x = self.signal(rng, 2, 65536, dtype, 0.0)
                check_storage_regression(block.run, frame_bytes=block.y.nbytes, repeats=5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("factor,n,ntaps", [(1, 3, 1), (3, 3, 2), (4, 4, 23), (8, 16, 97)])
def test_decimator_against_convolution_at_every_phase(dtype, factor, n, ntaps):
    rng = np.random.default_rng(32)
    x = rng.normal(size=(2, n * 5)).astype(dtype)
    taps = rng.normal(size=ntaps).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        x += 1j * rng.normal(size=x.shape)
        taps += 1j * rng.normal(size=taps.shape)
    for phase in range(factor):
        block = FirDecimate(taps=taps, factor=factor, phase=phase)
        block.setup(Ctx(rate=1000, n=n, channels=2, dtype=dtype))
        frames = []
        for chunk in np.split(x, 5, axis=1):
            block.x = chunk
            frames.append(block.run().copy())
        expected = np.stack([np.convolve(row, taps)[phase : x.shape[1] : factor] for row in x])
        bound = ntaps * np.finfo(dtype).eps * np.abs(x).max() * np.abs(taps).sum()
        np.testing.assert_allclose(np.concatenate(frames, axis=1), expected, rtol=0, atol=bound)
        block.reset()
        block.x = x[:, :n]
        np.testing.assert_array_equal(block.run(), frames[0])


@pytest.mark.parametrize("dtype,taps", [(np.int16, [1.0]), (np.float64, [1j])])
def test_decimator_rejects_unsupported_dtypes(dtype, taps):
    with pytest.raises(SetupError):
        FirDecimate(taps=np.array(taps), factor=1).setup(Ctx(rate=1, n=8, channels=1, dtype=dtype))


@pytest.mark.parametrize("kwargs", [{"factor": 0}, {"factor": True}, {"factor": 1.5}, {"taps": []}, {"taps": [np.nan]}])
def test_decimator_validation(kwargs):
    config = {"factor": 2, "taps": [1, 2, 1]} | kwargs
    with pytest.raises(SetupError):
        FirDecimate(**config).setup(Ctx(rate=1, n=8, channels=1))


@pytest.mark.parametrize("phase", [-1, 2, True, 0.5])
def test_bad_decimation_phase(phase):
    with pytest.raises(SetupError):
        FirDecimate(taps=[1], factor=2, phase=phase).setup(Ctx(rate=1, n=8, channels=1))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_decimator_storage(dtype):
    block = FirDecimate(taps=rrc(12, 8, 0.35), factor=8)
    block.setup(Ctx(rate=1, n=65536, channels=2, dtype=dtype))
    block.x = np.ones((2, 65536), dtype=dtype)
    check_storage_regression(block.run, frame_bytes=block.y.nbytes, repeats=3)
