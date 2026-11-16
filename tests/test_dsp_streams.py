from fractions import Fraction

import numpy as np
import pytest

from examples.blocks import DDC, MultiChannelDDC, Spectrum, Tone, Tuner
from graphion.dsp.blocks import ConstellationViewer, ErrorCounter, FirDecimate, Gain, Prbs, SymbolMapper
from graphion.dsp.communications import qpsk
from graphion.dsp.streams import stream_ctx
from graphion.profile.numpy_frame import Array, Block, C, Ctx, In, N, Out, SetupError


@pytest.mark.parametrize("rate", [1001, Fraction(1001, 3), 1001.0])
def test_rate_arithmetic(rate):
    block = FirDecimate(taps=np.ones(3), factor=4)
    block.setup(Ctx(rate=rate, n=16, channels=2))
    same = stream_ctx(block, "x")
    output = stream_ctx(block, "y")
    assert same.rate == rate and type(same.rate) is type(rate)
    assert output.rate == rate * Fraction(1, 4)
    assert isinstance(output.rate, float if isinstance(rate, float) else Fraction)
    assert output.n == 4 and output.channels == 2 and output.xp is np


def test_literal_dtype_and_source_reference():
    source = Prbs()
    source.setup(Ctx(rate=2, n=16, channels=2))
    assert stream_ctx(source, "y").dtype == np.uint8
    mapper = SymbolMapper(constellation=qpsk())
    mapper.setup(source.ctx)  # ctx dtype intentionally differs from literal bit ports.
    assert stream_ctx(mapper, "x").dtype == np.uint8
    assert stream_ctx(mapper, "y").dtype == np.complex128
    assert stream_ctx(mapper, "y").rate == 1


def test_before_setup():
    with pytest.raises(SetupError):
        stream_ctx(Gain(), "y")


@pytest.mark.parametrize(
    "block,port",
    [
        (Block(), "x"),
        (Gain(), "missing"),
        (Spectrum(), "psd"),
        (ErrorCounter(), "rate"),
        (ErrorCounter(), "errors"),
        (ErrorCounter(), "total"),
    ],
)
def test_outside_convention(block, port):
    block.setup(Ctx(rate=1, n=16, channels=1))
    with pytest.raises(SetupError, match="outside the stream-context convention"):
        stream_ctx(block, port)


def test_constellation_viewer_points_are_not_a_stream():
    viewer = ConstellationViewer(depth=8)
    viewer.setup(Ctx(rate=3, n=16, channels=2, dtype=np.complex64))
    assert stream_ctx(viewer, "x") == viewer.ctx
    with pytest.raises(SetupError, match="outside the stream-context convention"):
        stream_ctx(viewer, "points")


def test_invalid_shapes_and_reference_selection():
    class WrongReference(Block):
        x: Array = In(C, N)
        y: Array = Out(C, N / 2)
        stream_ports = ("y", "x")

    block = WrongReference()
    block.setup(Ctx(rate=1, n=8, channels=1))
    with pytest.raises(SetupError, match="reference stream shape"):
        stream_ctx(block, "x")

    class NotTwoDimensional(Block):
        x: Array = In(C, N)
        y: Array = Out(N)
        stream_ports = ("x", "y")

    other = NotTwoDimensional()
    other.setup(Ctx(rate=1, n=8, channels=1))
    with pytest.raises(SetupError, match="shapes"):
        stream_ctx(other, "y")


def test_forwarded_outputs_and_existing_conventions():
    for block, port, divisor, dtype in (
        (DDC(freq=0), "y", 16, np.complex128),
        (MultiChannelDDC(), "y2", 16, np.complex128),
        (Tone(), "y", 1, np.float64),
        (Tuner(freq=0), "y", 1, np.complex128),
        (Spectrum(), "x", 1, np.float64),
    ):
        block.setup(Ctx(rate=1024000, n=4096, channels=1))
        ctx = stream_ctx(block, port)
        assert ctx.rate == Fraction(1024000, divisor)
        assert ctx.n == 4096 // divisor and ctx.dtype == dtype
