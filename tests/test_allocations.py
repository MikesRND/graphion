"""Deterministic storage regressions, not machine-dependent timing thresholds."""

import unittest

import numpy as np

from graphion.dsp.blocks import Gain
from tests.allocation_helpers import (
    ALLOCATION_WORKLOADS,
    PEAK_ALLOWANCE_BYTES,
    check_block_storage,
    check_storage_regression,
    configured_block,
    sample_allocations,
)


class AllocationTests(unittest.TestCase):
    def test_reference_blocks_and_composite_reuse_data_buffers(self):
        for cls, channels, n, repeats in ALLOCATION_WORKLOADS:
            for dtype in (np.float32, np.float64, np.complex64, np.complex128):
                with self.subTest(block=cls.__name__, dtype=dtype):
                    block = configured_block(cls, dtype, channels=channels, n=n)
                    _, storage = check_block_storage(block, repeats=repeats)
                    self.assertTrue(storage["stable_identities"])
                    if cls.__name__ == "OnePoleLowpass":
                        self.assertIn("root._work", storage["buffers"])
                    if cls.__name__ == "GainChain":
                        self.assertEqual(storage["forwarded_outputs_without_boundary_storage"], ["root.y"])
                        self.assertEqual(storage["bytes"], 2 * block.y.nbytes)

    def test_freed_temporary_is_detected_even_when_snapshot_is_empty(self):
        class AllocatingGain(Gain):
            def output(self):
                self.y[...] = self.x * self.gain

        block = configured_block(AllocatingGain, np.float64)
        sample = sample_allocations(block.run)
        self.assertEqual(sample.retained_numpy_blocks, 0)
        self.assertGreaterEqual(sample.peak_extra_bytes, block.y.nbytes)
        with self.assertRaisesRegex(AssertionError, "peak allowance"):
            check_storage_regression(block.run, frame_bytes=block.y.nbytes)

    def test_retained_small_numpy_buffer_is_detected(self):
        class RetainingGain(Gain):
            def output(self):
                super().output()
                self.extra = np.empty(8, dtype=np.float64)

        block = configured_block(RetainingGain, np.float64)
        sample = sample_allocations(block.run)
        self.assertGreater(sample.retained_numpy_blocks, 0)
        self.assertGreaterEqual(sample.retained_numpy_bytes, 64)
        self.assertLess(sample.peak_extra_bytes, PEAK_ALLOWANCE_BYTES)
        with self.assertRaisesRegex(AssertionError, "survives processing"):
            check_storage_regression(block.run, frame_bytes=block.y.nbytes)
