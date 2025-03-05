import pytest
import numpy as np
from graphion.core import Tensor, Block


def test_tensor_initialization():
    data = np.random.randn(2, 3)
    tensor = Tensor(data)
    assert tensor._memory is not None
    assert np.array_equal(tensor._memory, data)
    assert tensor.shape == data.shape
    assert tensor.dtype == data.dtype


def test_tensor_empty_initialization():
    with pytest.raises(TypeError):
        Tensor()


def test_block1():
    b1 = Block()
    assert not b1.isInitialized
    b1.initialize()
    assert b1.isInitialized
    b1.terminate()
    assert not b1.isInitialized
    with pytest.raises(RuntimeError):
        b1.run()
