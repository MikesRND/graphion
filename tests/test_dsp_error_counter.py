import numpy as np
import pytest

from graphion.dsp.blocks import ErrorCounter
from tests.allocation_helpers import check_storage_regression
from tests.dsp_helpers import context


@pytest.mark.parametrize("skip", [0, 3, 9, 25])
def test_counter(skip):
    counter = ErrorCounter(skip=skip)
    counter.setup(context(5, 2, np.uint8))
    x = np.zeros((2, 20), dtype=np.uint8)
    ref = x.copy()
    ref[0, ::3] = 255  # generic inequality, not just bits
    ref[1, ::2] = 1
    for start in range(0, 20, 5):
        counter.x = x[:, start : start + 5]
        counter.ref = ref[:, start : start + 5]
        counter.run()
    np.testing.assert_array_equal(counter.errors, np.count_nonzero(ref[:, skip:], axis=1))
    np.testing.assert_array_equal(counter.total, [max(0, 20 - skip)] * 2)
    assert counter.total.dtype == counter.errors.dtype == np.uint64
    if skip < 20:
        np.testing.assert_allclose(counter.rate, np.count_nonzero(ref[:, skip:], axis=1) / (20 - skip))
    else:
        assert np.isnan(counter.rate).all()
    expected = tuple(value.copy() for value in (counter.total, counter.errors, counter.rate))
    counter.reset()
    for start in range(0, 20, 5):
        counter.x = x[:, start : start + 5]
        counter.ref = ref[:, start : start + 5]
        counter.run()
    for actual, replay in zip((counter.total, counter.errors, counter.rate), expected, strict=False):
        np.testing.assert_array_equal(actual, replay)


def test_counter_reset_preserves_output_storage_and_skip_run_defines_outputs():
    counter = ErrorCounter(skip=10)
    supplied = {
        "total": np.full(2, 123, dtype=np.uint64),
        "errors": np.full(2, 123, dtype=np.uint64),
        "rate": np.full(2, 123.0),
    }
    for name, values in supplied.items():
        counter.bind_output(name, values)
    counter.setup(context(5, 2, np.uint8))
    counter.x = np.zeros((2, 5), dtype=np.uint8)
    counter.ref = counter.x
    for _ in range(2):
        counter.reset()
        # Inspect caller-owned storage to enforce the no-output-writes reset rule.
        for values in supplied.values():
            np.testing.assert_array_equal(values, [123, 123])
        for _ in range(2):
            counter.run()
            np.testing.assert_array_equal(counter.total, [0, 0])
            np.testing.assert_array_equal(counter.errors, [0, 0])
            assert np.isnan(counter.rate).all()
            for values in supplied.values():
                values.fill(123)


def test_counter_storage():
    block = ErrorCounter(skip=100)
    block.setup(context(65536, 2, np.uint8))
    block.x = np.zeros((2, 65536), dtype=np.uint8)
    block.ref = block.x
    check_storage_regression(block.run, frame_bytes=65536, repeats=2)
