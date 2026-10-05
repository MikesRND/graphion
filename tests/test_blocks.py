import unittest
from unittest.mock import patch

import numpy as np

from graphion import (
    Array,
    Axis,
    Block,
    C,
    Control,
    Ctx,
    DeclarationError,
    DtypeError,
    In,
    N,
    Out,
    Param,
    ParamError,
    SetupError,
    ShapeError,
    State,
    StorageError,
    UnavailablePortError,
)
from graphion.dsp.blocks import FrameDelay, Gain, OnePoleLowpass
from graphion.profile.numpy_frame import _storage as arrays


def context(dtype=np.float64, n=4):
    return Ctx(rate=1e6, n=n, channels=2, dtype=dtype)


class BlockTests(unittest.TestCase):
    def test_gain_storage_and_borrowing(self):
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            with self.subTest(dtype=dtype):
                b = Gain(gain=2)
                b.setup(context(dtype))
                source = np.ones((2, 4), dtype=dtype)
                b.x = source
                output = b.y
                result = b.run()
                snapshot = result.copy()
                self.assertIsNot(result, output)
                self.assertTrue(np.shares_memory(result, output))
                self.assertTrue(output.flags.writeable)
                self.assertFalse(result.flags.writeable)
                self.assertFalse(b.x.flags.writeable)
                with self.assertRaises(ValueError):
                    result[...] = 0
                with self.assertRaises(ValueError):
                    b.x[...] = 0
                source.fill(3)
                self.assertIs(result, b.run())
                np.testing.assert_array_equal(result, 6)
                np.testing.assert_array_equal(snapshot, 2)
                b.reset()
                self.assertIs(output, b.y)
                self.assertIs(result, b.run())

    def test_input_rebinding_is_atomic_and_same_object_is_free(self):
        b = Gain()
        b.setup(context())
        raw = np.ones((2, 4))
        b.x = raw
        view = b.x
        with (
            patch.object(arrays, "validate", side_effect=AssertionError),
            patch.object(arrays, "readonly_view", side_effect=AssertionError),
            patch.object(arrays, "validate_input_storage", side_effect=AssertionError),
        ):
            b.x = raw
            raw.fill(7)
            np.testing.assert_array_equal(b.run(), 7)
        for invalid, error in (
            (np.ones((1, 4)), ShapeError),
            (np.ones((2, 4), dtype=np.float32), DtypeError),
            (b.y, StorageError),
            (b.run(), StorageError),
        ):
            with self.subTest(error=error):
                with self.assertRaises(error):
                    b.x = invalid
                self.assertIs(view, b.x)
        replacement = raw.copy()
        b.x = replacement
        self.assertIsNot(view, b.x)
        self.assertTrue(np.shares_memory(replacement, b.x))

    def test_prebound_input_checks_and_failed_setup(self):
        for raw, supplied, error in (
            (np.ones((1, 4)), None, ShapeError),
            (np.ones((2, 4), dtype=np.float32), None, DtypeError),
            (np.ones((2, 4)), True, StorageError),
        ):
            with self.subTest(error=error):
                b = Gain()
                b.x = raw
                if supplied:
                    b.bind_output("y", raw)
                with self.assertRaises(error):
                    b.setup(context())
                with self.assertRaises(SetupError):
                    b.run()
                with self.assertRaises(SetupError):
                    b.setup(context())

    def test_supplied_output_and_assignment_guards(self):
        b = Gain()
        supplied = np.empty((2, 4))
        b.bind_output("y", supplied)
        with patch.object(arrays, "allocate", side_effect=AssertionError):
            b.setup(context())
        self.assertIs(supplied, b.y)
        b.y.fill(1)
        b.y *= 2
        np.testing.assert_array_equal(supplied, 2)
        with self.assertRaises(StorageError):
            b.y = supplied.copy()
        with self.assertRaises(StorageError):
            b.bind_output("y", supplied)
        with self.assertRaises(StorageError):
            del b.y

    def test_supplied_output_requires_writability_and_exact_spec(self):
        readonly = np.empty((2, 4))
        readonly.flags.writeable = False
        for supplied, error in (
            (readonly, StorageError),
            (np.empty((2, 4), dtype=np.float32), DtypeError),
            (np.empty((8,)), ShapeError),
        ):
            b = Gain()
            b.bind_output("y", supplied)
            with self.assertRaises(error):
                b.setup(context())

    def test_two_outputs_disjoint_interleaving_and_overlap(self):
        class Pair(Block):
            a: Array = Out(N, dtype=float)
            b: Array = Out(N, dtype=float)

            def process(self):
                self.a.fill(1)
                self.b.fill(2)

        storage = np.empty(8)
        good = Pair()
        good.bind_output("a", storage[::2])
        good.bind_output("b", storage[1::2])
        good.setup(context())
        result = good.run()
        self.assertIs(result, good.run())
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(storage, [1, 2] * 4)
        for right in (storage[:4], storage[1:5]):
            bad = Pair()
            bad.bind_output("a", storage[:4])
            bad.bind_output("b", right)
            with self.assertRaisesRegex(StorageError, r"storage overlaps: root.*\.a \(Out\) and root.*\.b \(Out\)"):
                bad.setup(context())

    def test_writer_list_keeps_distinct_slots_and_work_limit_reason(self):
        shared = np.empty(4)
        writers = [arrays.Writer("a", shared, True), arrays.Writer("b", shared)]
        with self.assertRaisesRegex(StorageError, "storage overlaps"):
            arrays.validate_writers(writers)
        with patch.object(np, "shares_memory", side_effect=np.exceptions.TooHardError):
            with self.assertRaisesRegex(StorageError, "work limit"):
                arrays.validate_writers(writers)

    def test_configuration_is_declarative_and_arrays_are_copied(self):
        class Configured(Block):
            required: int = Param()
            coefficients: np.ndarray = Param(default=np.array([1.0, 2.0]))
            control: np.ndarray = Control(default=np.array([3.0]))

            def build(self):
                self.built_with = self.required

        with self.assertRaises(ParamError):
            Configured()
        with self.assertRaises(ParamError):
            Configured(required=1, typo=3)
        a, b = Configured(required=1), Configured(required=2)
        self.assertEqual(a.built_with, 1)
        self.assertFalse(np.shares_memory(a.coefficients, b.coefficients))
        self.assertFalse(a.coefficients.flags.writeable)
        with self.assertRaises(ParamError):
            a.required = 2
        with self.assertRaises(ValueError):
            a.coefficients[0] = 7
        supplied = np.array([5.0])
        a.control = supplied
        supplied.fill(9)
        np.testing.assert_array_equal(a.control, [5.0])
        np.testing.assert_array_equal(b.control, [3.0])

    def test_state_defaults_and_reset_are_in_place(self):
        class Stateful(Block):
            scalar: int = State(default=3)
            memory: Array = State(2, default=np.array([1.0, 2.0]), dtype=float)
            y: Array = Out(dtype=float)

            def reset(self):
                self.scalar += 1

            def process(self):
                self.y[...] = self.scalar + self.memory.sum()

        a, b = Stateful(), Stateful()
        a.setup(context())
        b.setup(context())
        state, output, result = a.memory, a.y, a.run()
        self.assertEqual(result.shape, ())
        a.scalar = 99
        a.memory.fill(10)
        np.testing.assert_array_equal(b.memory, [1.0, 2.0])
        with self.assertRaises(StorageError):
            a.memory = a.memory.copy()
        with self.assertRaises(StorageError):
            # An external input into this block is tested separately below.
            arrays.validate_input_storage(a.memory, "input", a._writers)
        a.reset()
        self.assertIs(state, a.memory)
        self.assertIs(output, a.y)
        self.assertIs(result, a.run())
        self.assertEqual(a.scalar, 4)
        np.testing.assert_array_equal(a.memory, [1.0, 2.0])

    def test_context_dependent_state_reset_and_bad_array_default(self):
        class History(Block):
            x: Array = In(C, N)
            memory: Array = State(C, N, default=0)

            def reset(self):
                self.memory[:] = np.arange(self.ctx.n)

        b = History()
        b.setup(context())
        np.testing.assert_array_equal(b.memory, np.tile(np.arange(4), (2, 1)))
        with self.assertRaises(StorageError):
            b.x = b.memory

        class Invalid(Block):
            memory: Array = State(N, default=np.ones(3), dtype=float)

        with self.assertRaises(ShapeError):
            Invalid().setup(context())

    def test_lifecycle_inheritance_calls_hooks_once(self):
        class Parent(Block):
            p: int = Param(default=1)
            count: int = State(default=0)

            def build(self):
                self.events = []

            def setup(self, ctx):
                self.events.append("parent setup")

            def reset(self):
                self.count += 1
                self.events.append("parent reset")

            def process(self):
                self.count += 10

        class Child(Parent):
            p: int = Param(default=2)

            def setup(self, ctx):
                super().setup(ctx)
                self.events.append("child setup")

            def reset(self):
                super().reset()
                self.count += 1
                self.events.append("child reset")

        b = Child()
        self.assertEqual(b.p, 2)
        b.setup(context())
        self.assertEqual(b.events, ["parent setup", "child setup", "parent reset", "child reset"])
        self.assertEqual(b.count, 2)
        self.assertIsNone(b.run())
        self.assertEqual(b.count, 12)
        b.reset()
        self.assertEqual(b.count, 2)

    def test_scalar_state_defaults_report_dtype_errors(self):
        for default, dtype in (
            (1j, np.float64),
            (np.complex64(1j), np.float32),
            (np.complex128(1j), np.int64),
            ("not a number", np.float64),
            (1000, np.int8),
        ):
            with self.subTest(default=default, dtype=dtype):

                class Invalid(Block):
                    memory: Array = State(C, default=default)

                with self.assertRaisesRegex(DtypeError, "memory.*default"):
                    Invalid().setup(context(dtype))

    def test_scalar_state_default_conversions_still_work(self):
        class Real(Block):
            memory: Array = State(C, default=1.25)

        class Complex(Block):
            memory: Array = State(C, default=1 + 2j)

        for block, dtype, expected in (
            (Real(), np.float32, 1.25),
            (Real(), np.complex64, 1.25 + 0j),
            (Complex(), np.complex64, 1 + 2j),
        ):
            block.setup(context(dtype))
            np.testing.assert_array_equal(block.memory, expected)
            block.memory.fill(0)
            block.reset()
            np.testing.assert_array_equal(block.memory, expected)

    def test_framework_owned_names_cannot_be_overridden(self):
        for name in (
            "__init__",
            "run",
            "run_output",
            "run_update",
            "bind_output",
            "schema",
            "ctx",
            "child",
            "connect",
            "children",
            "connections",
            "describe",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(DeclarationError, "framework-owned"):
                    type("InvalidOverride", (Block,), {name: lambda self: None})

    def test_declared_framework_names_are_explicitly_reserved(self):
        for name in (
            "run",
            "bind_output",
            "schema",
            "ctx",
            "build",
            "setup",
            "reset",
            "process",
            "output",
            "update",
            "run_output",
            "run_update",
            "child",
            "connect",
            "children",
            "connections",
            "describe",
            "_private",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(DeclarationError, "reserved declaration"):
                    type("InvalidDeclaration", (Block,), {"__annotations__": {name: int}, name: Param()})

    def test_readiness_and_namespace_check_precedes_author_setup(self):
        class Probe(Gain):
            def setup(self, ctx):
                self.body_ran = True

        b = Probe()
        with self.assertRaises(SetupError):
            b.reset()
        with self.assertRaises(SetupError):
            b.run()
        with self.assertRaises(SetupError):
            b.setup(context().replace(xp=object()))
        self.assertFalse(hasattr(b, "body_ran"))
        good = Gain()
        good.setup(context())
        with self.assertRaises(UnavailablePortError):
            good.run()
        with self.assertRaises(SetupError):
            good.setup(context())

    def test_axis_and_invalid_context(self):
        class Shaped(Block):
            width: int = Param(default=3)
            y: Array = Out(C, Axis("width"))

        b = Shaped()
        b.setup(context(np.complex64))
        self.assertEqual(b.y.shape, (2, 3))
        self.assertEqual(b.y.dtype, np.complex64)
        for changes in ({"rate": 0}, {"n": True}, {"channels": -1}, {"dtype": object}):
            with self.assertRaises(SetupError):
                context().replace(**changes)

    def test_invalid_declarations_fail_early(self):
        with self.assertRaises(DeclarationError):

            class CustomConstructor(Block):
                def __init__(self):
                    pass

        with self.assertRaises(DeclarationError):

            class Reserved(Block):
                ctx: int = Param(default=1)

        with self.assertRaises(DeclarationError):

            class MissingDefault(Block):
                memory: Array = State(N, dtype=float)

        with self.assertRaises(DeclarationError):

            class ChangedKind(Gain):
                gain: float = Param(default=2)

    def test_frame_delay_call_semantics_and_reset(self):
        b = FrameDelay()
        b.setup(context(np.complex64))
        b.x = np.full((2, 4), 1 + 2j, dtype=np.complex64)
        np.testing.assert_array_equal(b.run(), 0)
        b.x = np.full((2, 4), 3 + 4j, dtype=np.complex64)
        np.testing.assert_array_equal(b.run(), 1 + 2j)
        held = b.held
        b.reset()
        self.assertIs(held, b.held)
        np.testing.assert_array_equal(b.run(), 0)

    def test_one_pole_lowpass_frame_continuity_and_reset(self):
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            with self.subTest(dtype=dtype):
                signal = np.arange(16, dtype=np.float64).reshape(2, 8).astype(dtype)
                if np.issubdtype(dtype, np.complexfloating):
                    signal += 1j
                expected = np.empty_like(signal)
                history = np.zeros(2, dtype=dtype)
                for i in range(8):
                    history = 0.75 * history + 0.25 * signal[:, i]
                    expected[:, i] = history
                b = OnePoleLowpass(pole=0.75)
                b.setup(context(dtype))
                actual = []
                for frame in (signal[:, :4], signal[:, 4:]):
                    b.x = frame
                    actual.append(b.run().copy())
                np.testing.assert_allclose(np.concatenate(actual, axis=1), expected, rtol=1e-6)
                b.reset()
                b.x = signal[:, :4]
                np.testing.assert_allclose(b.run(), expected[:, :4], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
