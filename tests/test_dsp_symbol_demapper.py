import numpy as np
import pytest

from graphion.dsp.blocks import SymbolDemapper, SymbolMapper
from graphion.dsp.communications import qpsk
from graphion.dsp.streams import stream_ctx
from graphion.profile.numpy_frame import SetupError
from tests.allocation_helpers import check_storage_regression
from tests.dsp_helpers import constellation_tables, context, label_bits


@pytest.mark.parametrize("table", constellation_tables())
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_demapper_inverts_mapper_labels(table, dtype):
    bits = label_bits(table)
    mapper = SymbolMapper(constellation=table)  # The mapper only supplies the symbols and their context.
    mapper.setup(context(bits.size))
    mapper.x = bits
    mapper.run()
    demapper = SymbolDemapper(constellation=table)
    demapper.setup(stream_ctx(mapper, "y").replace(dtype=dtype))
    demapper.x = mapper.y.astype(dtype)
    np.testing.assert_array_equal(demapper.run(), bits)
    demapper.reset()
    np.testing.assert_array_equal(demapper.run(), bits)


@pytest.mark.parametrize("table", [[], [1], [1, 2, 3], [1, 1], [1, np.inf], [[1, -1]]])
def test_demapper_bad_constellation(table):
    with pytest.raises(SetupError):
        SymbolDemapper(constellation=table).setup(context(dtype=np.complex128))


def test_demapper_rejects_real_samples():
    with pytest.raises(SetupError):
        SymbolDemapper(constellation=qpsk()).setup(context())


def test_demapper_ties():
    block = SymbolDemapper(constellation=np.array([1, -1, 1j, -1j]))
    block.setup(context(3, dtype=np.complex128))
    block.x = np.array([[0, 0.5 + 0.5j, -0.5 - 0.5j]])
    np.testing.assert_array_equal(block.run(), [[0, 0, 0, 0, 0, 1]])


def test_demapper_storage():
    block = SymbolDemapper(constellation=qpsk())
    block.setup(context(65536, 2, np.complex128))
    block.x = np.zeros((2, 65536), dtype=np.complex128)
    check_storage_regression(block.run, frame_bytes=65536, repeats=2)
