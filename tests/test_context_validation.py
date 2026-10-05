"""Numeric Implementation Profile boundaries beyond the example block workloads."""

import json
import unittest
from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
from numbers import Real
from unittest.mock import patch

import numpy as np

from graphion import (
    Array,
    Axis,
    Block,
    C,
    Ctx,
    DtypeError,
    In,
    N,
    Out,
    Param,
    SetupError,
    ShapeError,
    State,
    StorageError,
)
from graphion.profile.numpy_frame import _storage as arrays


def context(**changes):
    return Ctx(rate=1e6, n=4, channels=2).replace(**changes)


class Copy(Block):
    x: Array = In(C, N)
    y: Array = Out(C, N)

    def process(self):
        np.copyto(self.y, self.x, casting="no")


class ContextValidationTests(unittest.TestCase):
    def test_context_is_immutable_and_replace_preserves_other_fields(self):
        original = Ctx(rate=np.float64(12.5), n=np.int64(4), channels=np.int32(2), dtype="complex64")
        derived = original.replace(rate=25, n=8)
        self.assertIsNot(original, derived)
        self.assertEqual((original.rate, original.n), (12.5, 4))
        self.assertEqual((derived.rate, derived.n, derived.channels), (25, 8, 2))
        self.assertEqual(derived.dtype, np.dtype(np.complex64))
        self.assertIs(derived.xp, np)
        marker = object()
        alternate = original.replace(xp=marker)
        self.assertIs(alternate.replace(n=8).xp, marker)
        self.assertIs(alternate.replace(xp=np).xp, np)
        for name in ("rate", "n", "channels", "dtype", "xp"):
            with self.subTest(field=name), self.assertRaises(FrozenInstanceError):
                setattr(original, name, None)
        with self.assertRaises(TypeError):
            original.replace(unknown=1)

    def test_invalid_context_numbers_raise_setup_errors(self):
        cases = {
            "rate": [0, -1, float("nan"), float("inf"), float("-inf"), True, np.bool_(True), 1j, "1", None],
            "n": [0, -1, True, np.bool_(True), 1.0, np.float64(1), "1", None],
            "channels": [0, -1, True, np.bool_(True), 1.0, "1", None],
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(field=name, value=value), self.assertRaisesRegex(SetupError, name):
                    context(**{name: value})

    def test_supported_rates_preserve_exact_values_and_canonicalize_numpy_scalars(self):
        rates = [
            (1, int),
            (1.0, float),
            (np.int64(3), int),
            (np.uint64(2**63), int),
            (np.float32(1.1), float),
            (np.float64(3.5), float),
            (np.longdouble("1.1"), float),
            (Fraction(7, 3), Fraction),
            (Fraction(10**1000 + 1, 10**1000), Fraction),
            (2**80 + 1, int),
        ]
        for rate, expected_type in rates:
            with self.subTest(rate_type=type(rate)):
                configured = context(rate=rate)
                self.assertIs(type(configured.rate), expected_type)
                expected = float(rate) if expected_type is float else rate
                self.assertEqual(configured.rate, expected)
                self.assertEqual(configured.replace(n=8).rate, configured.rate)
                derived = configured.replace(rate=Fraction(1, 3))
                self.assertIs(type(derived.rate), Fraction)
                self.assertEqual(derived.rate, Fraction(1, 3))
                self.assertEqual(configured.rate, expected)
        exact = 2**80 + 1
        self.assertEqual(context(rate=exact).rate, exact)
        self.assertNotEqual(context(rate=exact).rate, float(exact))

    def test_fractional_rate_division_through_replace_stays_exact(self):
        original = context(rate=Fraction(48_000_000, 1001))
        derived = original.replace(rate=original.rate / 4, n=1)
        self.assertIs(type(original.rate), Fraction)
        self.assertIs(type(derived.rate), Fraction)
        self.assertEqual(derived.rate, Fraction(12_000_000, 1001))
        self.assertEqual(original.rate, Fraction(48_000_000, 1001))
        self.assertEqual(derived.replace(channels=3).rate, derived.rate)

    def test_rate_types_exclude_decimal_custom_floatable_and_registered_real(self):
        class Floatable:
            def __float__(self):
                raise AssertionError("unsupported rate must not be converted")

        @Real.register
        class RegisteredReal(Floatable):
            pass

        self.assertIsInstance(RegisteredReal(), Real)
        for rate in (Decimal("1.5"), Floatable(), RegisteredReal(), True, np.bool_(True)):
            with self.subTest(rate_type=type(rate)), self.assertRaisesRegex(SetupError, "rate"):
                context(rate=rate)

    def test_rate_float_approximation_rejects_overflow_and_underflow_to_zero(self):
        smallest = float.fromhex("0x0.0000000000001p-1022")
        for rate in (smallest, Fraction(1, 2**1074)):
            configured = context(rate=rate)
            self.assertEqual(configured.rate, rate)
            self.assertEqual(float(configured.rate), smallest)
        invalid = [
            10**1000,
            Fraction(1, 10**1000),
            Fraction(1, 2**1075),
            Fraction(10**1000, 3),
            np.longdouble("nan"),
            np.longdouble("inf"),
            np.longdouble("-inf"),
        ]
        extended = np.finfo(np.longdouble)
        if extended.max > np.finfo(np.float64).max:
            invalid.append(extended.max)
        if extended.tiny < np.finfo(np.float64).smallest_subnormal:
            invalid.append(extended.tiny)
        for rate in invalid:
            with self.subTest(rate=rate), self.assertRaisesRegex(SetupError, "rate"):
                context(rate=rate)

    def test_context_rate_descriptions_preserve_exact_and_approximate_json_values(self):
        for rate in (
            1,
            1.0,
            Fraction(7, 3),
            Fraction(48_000_000, 1001),
            Fraction(10**1000 + 1, 10**1000),
            np.float32(1.1),
            2**80 + 1,
        ):
            with self.subTest(rate_type=type(rate)):
                block = Block()
                block.setup(context(rate=rate))
                description = block.describe()
                exported = description["blocks"][0]["context"]
                exported_rate = exported["rate"]
                self.assertIs(type(exported_rate), float)
                self.assertEqual(exported_rate, float(block.ctx.rate))
                if isinstance(block.ctx.rate, (int, Fraction)):
                    numerator, denominator = block.ctx.rate.as_integer_ratio()
                    exact = {"numerator": numerator, "denominator": denominator}
                else:
                    exact = None
                self.assertEqual(exported["rate_exact"], exact)
                decoded = json.loads(json.dumps(description, allow_nan=False))
                self.assertEqual(decoded["blocks"][0]["context"], exported)

    def test_malformed_and_nonnumeric_dtypes_raise_framework_errors(self):
        selectors = [
            object,
            "U2",
            "S2",
            "datetime64[ns]",
            "timedelta64[ns]",
            [("value", "f8")],
            "not-a-dtype",
            {"names": ["x"], "formats": []},
            ("f8", (-1,)),
            ("f8", (10**100,)),
        ]
        for selector in selectors:
            with self.subTest(dtype=selector):
                with self.assertRaises(SetupError):
                    context(dtype=selector)

                class Invalid(Block):
                    y: Array = Out(N, dtype=selector)

                with self.assertRaisesRegex(DtypeError, "y"):
                    Invalid().setup(context())

    def test_literal_dtype_mappings_and_sample_dtype_are_exact(self):
        for selector, expected in (
            (bool, np.bool_),
            (int, np.int64),
            (float, np.float64),
            (complex, np.complex128),
            (np.uint16, np.uint16),
            (np.float32, np.float32),
            (np.complex64, np.complex64),
        ):
            with self.subTest(dtype=selector):

                class Explicit(Block):
                    literal: Array = Out(2, dtype=selector)
                    sample: Array = Out(2)
                    memory: Array = State(2, default=1, dtype=selector)

                block = Explicit()
                block.setup(context(dtype=np.complex64))
                self.assertEqual(block.literal.dtype, np.dtype(expected))
                self.assertEqual(block.memory.dtype, np.dtype(expected))
                self.assertEqual(block.sample.dtype, np.dtype(np.complex64))
                np.testing.assert_array_equal(block.memory, np.ones(2, dtype=expected))

    def test_all_numeric_signal_kinds_preserve_dtype_without_casting(self):
        for dtype in (np.bool_, np.int16, np.uint32, np.float32, np.complex64):
            with self.subTest(dtype=dtype):
                block = Copy()
                block.setup(context(dtype=dtype))
                source = np.ones((2, 4), dtype=dtype)
                block.x = source
                result = block.run()
                np.testing.assert_array_equal(result, source)
                self.assertEqual(result.dtype, source.dtype)
                self.assertTrue(np.shares_memory(block.x, source))
                self.assertTrue(source.flags.writeable)
                self.assertFalse(block.x.flags.writeable)
                self.assertFalse(result.flags.writeable)

    def test_literal_and_named_axes_reject_invalid_sizes(self):
        for size in (-1, np.int64(-1), True, np.bool_(True), 1.0, np.float64(2), None, "2"):
            with self.subTest(size=size):

                class Literal(Block):
                    y: Array = Out(size)

                class Named(Block):
                    width: object = Param(default=size)
                    y: Array = Out(Axis("width"))

                for cls in (Literal, Named):
                    with self.assertRaisesRegex(SetupError, "y.*axis"):
                        cls().setup(context())
        for name in (None, 1, [], "missing"):

            class Missing(Block):
                y: Array = Out(Axis(name))

            with self.subTest(name=name), self.assertRaisesRegex(SetupError, "y.*axis"):
                Missing().setup(context())

    def test_zero_and_numpy_integer_axes_resolve_once_in_memory_order(self):
        class Shaped(Block):
            y: Array = Out(np.int32(3), C, Axis("width"), N)
            empty: Array = Out(0, N)

            def setup(self, ctx):
                self.width = np.int64(5)

        block = Shaped()
        block.setup(context())
        self.assertEqual(block.y.shape, (3, 2, 5, 4))
        self.assertEqual(block.empty.shape, (0, 4))
        storage = block.y
        block.width = 99
        block.reset()
        self.assertIs(block.y, storage)
        self.assertEqual(block.y.shape, (3, 2, 5, 4))

    def test_signal_validation_rejects_nonarrays_without_conversion(self):
        class Convertible:
            def __array__(self, *args, **kwargs):
                raise AssertionError("signal conversion must not be invoked")

        for value in ([[1.0] * 4] * 2, memoryview(bytearray(64)), np.float64(1), Convertible()):
            block = Copy()
            block.setup(context())
            original = np.ones((2, 4))
            block.x = original
            reader = block.x
            with self.subTest(value_type=type(value)):
                with self.assertRaises(DtypeError):
                    block.x = value
                self.assertIs(block.x, reader)
                with self.assertRaises(DtypeError):
                    Copy().bind_output("y", value)

    def test_signal_validation_rejects_unsupported_dtypes_and_shape_without_rebinding(self):
        block = Copy()
        block.setup(context())
        original = np.ones((2, 4))
        block.x = original
        reader = block.x
        for dtype in (object, "U1", "S1", "datetime64[ns]", "timedelta64[ns]", [("value", "f8")], np.float32):
            invalid = np.empty((2, 4), dtype=dtype)
            with self.subTest(dtype=dtype):
                with self.assertRaises(DtypeError):
                    block.x = invalid
                self.assertIs(block.x, reader)
                supplied = Copy()
                supplied.bind_output("y", invalid)
                with self.assertRaises(DtypeError):
                    supplied.setup(context())
                self.assertTrue(invalid.flags.writeable)
        with self.assertRaises(ShapeError):
            block.x = np.ones((4, 2))
        self.assertIs(block.x, reader)
        np.testing.assert_array_equal(block.run(), original)

    def test_complex_array_state_defaults_require_exact_dtype_and_shape(self):
        declared = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        declared.flags.writeable = False

        class ComplexState(Block):
            memory: Array = State(2, default=declared, dtype=np.complex64)

        first, second = ComplexState(), ComplexState()
        first.setup(context())
        second.setup(context())
        self.assertFalse(np.shares_memory(first.memory, declared))
        self.assertFalse(np.shares_memory(first.memory, second.memory))
        self.assertFalse(declared.flags.writeable)
        storage = first.memory
        first.memory.fill(0)
        first.reset()
        self.assertIs(first.memory, storage)
        np.testing.assert_array_equal(first.memory, declared)
        np.testing.assert_array_equal(second.memory, declared)
        for shape, dtype, error in (
            (2, np.float32, DtypeError),
            (2, np.complex128, DtypeError),
            (3, np.complex64, ShapeError),
        ):

            class InvalidState(Block):
                memory: Array = State(shape, default=declared, dtype=dtype)

            with self.subTest(shape=shape, dtype=dtype), self.assertRaises(error):
                InvalidState().setup(context())

    def test_unknown_output_writability_is_rejected_without_changing_flags(self):
        for known in (None, False):
            output = np.empty((2, 4))
            block = Copy()
            block.bind_output("y", output)
            with self.subTest(known=known), patch.object(arrays, "is_writable", return_value=known):
                with self.assertRaisesRegex(StorageError, "known writable"):
                    block.setup(context())
            self.assertTrue(output.flags.writeable)
        readonly = np.empty((2, 4))
        readonly.flags.writeable = False
        block = Copy()
        block.bind_output("y", readonly)
        with self.assertRaises(StorageError):
            block.setup(context())
        self.assertFalse(readonly.flags.writeable)

    def test_borrowing_preserves_input_and_supplied_output_flags_through_run_and_reset(self):
        output = np.empty((2, 4))
        block = Copy()
        block.bind_output("y", output)
        block.setup(context())
        for writable in (True, False):
            source = np.arange(8, dtype=np.float64).reshape(2, 4)
            source.flags.writeable = writable
            block.x = source
            reader, exported = block.x, block.run()
            for _ in range(2):
                block.reset()
                self.assertIs(block.run(), exported)
                self.assertIs(block.x, reader)
                np.testing.assert_array_equal(exported, source)
                self.assertIs(bool(source.flags.writeable), writable)
                self.assertTrue(output.flags.writeable)
                self.assertFalse(reader.flags.writeable)
                self.assertFalse(exported.flags.writeable)
