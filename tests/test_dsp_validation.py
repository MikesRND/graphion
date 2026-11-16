import numpy as np
import pytest

from graphion.dsp._validation import is_integer


@pytest.mark.parametrize("value", [0, 7, -3, np.int8(4), np.uint64(9), np.int64(-1)])
def test_integers_of_either_kind_are_accepted(value):
    assert is_integer(value)


@pytest.mark.parametrize("value", [True, False, np.True_, 1.0, np.float64(2), "3", None, 1 + 0j, [1]])
def test_bools_and_other_types_are_rejected(value):
    assert not is_integer(value)
    assert not is_integer(value, 0)


def test_minimum_is_inclusive():
    assert is_integer(3, 3) and is_integer(np.int64(4), 3)
    assert not is_integer(2, 3) and not is_integer(np.int64(-1), 0)
