import numpy as np
import pytest

from graphion.dsp.blocks import SymbolMapper
from graphion.dsp.communications import qpsk
from graphion.profile.numpy_frame import Block, SetupError
from tests.allocation_helpers import check_storage_regression
from tests.dsp_helpers import constellation_tables, context, label_bits


@pytest.mark.parametrize("table", constellation_tables())
def test_mapper_labels(table):
    bits = label_bits(table)
    mapper = SymbolMapper(constellation=table)
    mapper.setup(context(bits.size))
    mapper.x = bits
    np.testing.assert_array_equal(mapper.run()[0], table)
    assert mapper.y.dtype == np.complex128
    mapper.reset()
    np.testing.assert_array_equal(mapper.run()[0], table)


@pytest.mark.parametrize("table", [[], [1], [1, 2, 3], [1, 1], [1, np.inf], [[1, -1]]])
def test_mapper_bad_constellation(table):
    with pytest.raises(SetupError):
        SymbolMapper(constellation=table).setup(context(dtype=np.complex128))


def test_mapper_frame_divisibility():
    with pytest.raises(SetupError):
        SymbolMapper(constellation=qpsk()).setup(context(3))


@pytest.mark.parametrize("value", [2, 128, 255])
def test_invalid_bits_leave_output_unwritten_and_require_reset(value):
    class Composite(Block):
        def build(self):
            self.mapper = self.child("mapper", SymbolMapper(constellation=qpsk()))

        def process(self):
            self.mapper.run()

    root = Composite()
    root.setup(context(4))
    output = root.mapper.y
    root.mapper.x = np.zeros((1, 4), dtype=np.uint8)
    root.run()
    before = output.copy()
    root.mapper.x = np.array([[0, 1, 1, value]], dtype=np.uint8)
    with pytest.raises(ValueError, match="unpacked bits"):
        root.run()
    np.testing.assert_array_equal(output, before)
    for block in (root, root.mapper):
        with pytest.raises(Exception, match="reset"):
            block.run()
    root.reset()
    root.mapper.x = np.ones((1, 4), dtype=np.uint8)
    root.run()
    np.testing.assert_array_equal(root.mapper.y, [[qpsk()[3], qpsk()[3]]])


def test_mapper_storage():
    block = SymbolMapper(constellation=qpsk())
    block.setup(context(65536, 2, np.uint8))
    block.x = np.zeros((2, 65536), dtype=np.uint8)
    check_storage_regression(block.run, frame_bytes=65536, repeats=2)
