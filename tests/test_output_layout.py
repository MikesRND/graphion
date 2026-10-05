import unittest
from unittest.mock import patch

import numpy as np

from graphion import (
    Array,
    Block,
    Ctx,
    DtypeError,
    In,
    Out,
    SetupError,
    ShapeError,
    StorageError,
)
from graphion.dsp.blocks import Gain
from graphion.profile.numpy_frame import _storage as arrays


def context(shape):
    return Ctx(rate=1, channels=shape[0], n=shape[1], dtype=np.float64)


class OutputLayoutTests(unittest.TestCase):
    def test_zero_stride_gain_is_rejected_before_processing(self):
        storage = np.array([123.0])
        output = np.ndarray((1, 4), dtype=np.float64, buffer=storage, strides=(0, 0))
        block = Gain(gain=2)
        block.x = np.array([[1.0, 2.0, 3.0, 4.0]])
        block.bind_output("y", output)
        with patch.object(block, "output", side_effect=AssertionError("must not run")):
            with self.assertRaisesRegex(StorageError, "y: unsupported output layout"):
                block.setup(context(output.shape))
            with self.assertRaises(SetupError):
                block.run()
        np.testing.assert_array_equal(storage, [123.0])

    def test_overlapping_and_conservative_layouts_are_rejected(self):
        for shape, strides in (
            ((3, 2), (8, 16)),
            ((2, 2), (8, 4)),
            ((3, 2), (16, 24)),  # Disjoint elements, but not nested spans.
        ):
            with self.subTest(shape=shape, strides=strides):
                output = np.ndarray(shape, dtype=np.float64, buffer=np.empty(128, dtype=np.uint8), strides=strides)
                block = Gain()
                block.bind_output("y", output)
                with self.assertRaisesRegex(StorageError, "y: unsupported output layout"):
                    block.setup(context(shape))

    def test_supported_layouts_compute_in_the_supplied_buffer(self):
        complex_storage = np.full((2, 4), 7 + 11j, dtype=np.complex128)
        outputs = {
            "contiguous": np.empty((2, 4)),
            "fortran": np.empty((2, 4), order="F"),
            "transpose": np.empty((4, 2)).T,
            "reversed": np.empty((2, 4))[:, ::-1],
            "stepped": np.empty((4, 8))[::2, ::2],
            "real": complex_storage.real,
            "imaginary": complex_storage.imag,
            "singleton_zero_stride": np.ndarray((1, 4), dtype=np.float64, buffer=np.empty(4), strides=(0, 8)),
        }
        for label, output in outputs.items():
            with self.subTest(layout=label):
                source = np.arange(output.size, dtype=np.float64).reshape(output.shape)
                block = Gain(gain=2)
                block.bind_output("y", output)
                block.setup(context(output.shape))
                block.x = source
                self.assertIs(block.y, output)
                # Layout work belongs to setup, not processing or reset.
                with patch.object(arrays, "validate_output_layout", side_effect=AssertionError):
                    result = block.run()
                    np.testing.assert_array_equal(result, source * 2)
                    np.testing.assert_array_equal(output, source * 2)
                    self.assertTrue(np.shares_memory(result, output))
                    self.assertFalse(result.flags.writeable)
                    block.reset()
                    self.assertIs(result, block.run())

    def test_empty_output_accepts_zero_strides(self):
        class Empty(Block):
            x: Array = In(1, 0, dtype=float)
            y: Array = Out(1, 0, dtype=float)

            def process(self):
                np.multiply(self.x, 2, out=self.y)

        output = np.ndarray((1, 0), dtype=np.float64, buffer=np.empty(1), strides=(0, 0))
        block = Empty()
        block.bind_output("y", output)
        block.setup(context((1, 1)))
        block.x = np.empty((1, 0))
        self.assertIs(block.y, output)
        np.testing.assert_array_equal(block.run(), np.empty((1, 0)))
        self.assertFalse(block.run().flags.writeable)

        readonly = output.view()
        readonly.flags.writeable = False
        for invalid, error in (
            (readonly, StorageError),
            (np.empty((1, 0), dtype=np.float32), DtypeError),
            (np.empty((0, 1)), ShapeError),
        ):
            with self.subTest(error=error):
                bad = Empty()
                bad.bind_output("y", invalid)
                with self.assertRaises(error):
                    bad.setup(context((1, 1)))

    def test_zero_dimensional_output_is_supported(self):
        class ScalarOutput(Block):
            y: Array = Out(dtype=float)

            def process(self):
                self.y[...] = 6

        output = np.empty(())
        block = ScalarOutput()
        block.bind_output("y", output)
        block.setup(context((1, 1)))
        self.assertIs(block.y, output)
        self.assertEqual(block.run().item(), 6)

    def test_input_self_overlap_does_not_require_output_layout(self):
        for shape, strides in (((1, 4), (0, 0)), ((3, 2), (8, 16)), ((2, 2), (8, 4)), ((3, 2), (16, 24))):
            with self.subTest(shape=shape, strides=strides):
                source = np.ndarray(shape, dtype=np.float64, buffer=np.arange(16, dtype=np.float64), strides=strides)
                source.flags.writeable = False
                expected = source.copy() * 2
                block = Gain(gain=2)
                with patch.object(arrays, "validate_output_layout", side_effect=AssertionError):
                    block.x = source
                    block.setup(context(shape))
                    np.testing.assert_array_equal(block.run(), expected)
                    block.x = source.copy()
                    np.testing.assert_array_equal(block.run(), expected)


if __name__ == "__main__":
    unittest.main()
