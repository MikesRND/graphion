"""Execution bookkeeping, author scratch ownership and failure recovery."""

import inspect
import unittest
from collections.abc import MutableMapping
from unittest.mock import patch

import numpy as np

from graphion import Array, Block, Control, Ctx, ExecutionError, In, N, Out, SetupError, State
from graphion.profile.numpy_frame import _storage as arrays


def context():
    return Ctx(rate=1, n=4, channels=1)


class LookupOnly(MutableMapping):
    """Allow cached field lookup while rejecting collection traversal/copying."""

    def __init__(self, values):
        self.values_by_name = values

    def __getitem__(self, key):
        return self.values_by_name[key]

    def __setitem__(self, key, value):
        self.values_by_name[key] = value

    def __delitem__(self, key):
        raise AssertionError("execution deleted a cached binding")

    def __iter__(self):
        raise AssertionError("execution traversed cached bindings")

    def __len__(self):
        raise AssertionError("execution collected cached bindings")


class ExecutionContractTests(unittest.TestCase):
    def test_run_uses_cached_bindings_and_results_without_collection_building(self):
        class Pair(Block):
            x: Array = In(N)
            plus: Array = Out(N)
            minus: Array = Out(N)

            def build(self):
                self.calls = 0

            def process(self):
                self.calls += 1
                np.add(self.x, 1, out=self.plus)
                np.subtract(self.x, 1, out=self.minus)

        model = Pair()
        model.setup(context())
        source = np.arange(4, dtype=np.float64)
        model.x = source
        result = model.run()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(tuple(inspect.signature(Block.run).parameters), ("self",))
        # This guards collection from input/schema traversal and cached returns;
        # it is not a general heap-allocation detector for author numerical code.
        bindings = model._values
        model._values = LookupOnly(bindings)
        try:
            with (
                patch.object(Pair, "schema", LookupOnly(Pair.schema)),
                patch.object(arrays, "validate", side_effect=AssertionError("per-run validation")),
                patch.object(arrays, "readonly_view", side_effect=AssertionError("per-run export")),
            ):
                for value in (2, 5, 9):
                    source.fill(value)
                    current = model.run()
                    self.assertIs(current, result)
                    self.assertIs(current[0], result[0])
                    self.assertIs(current[1], result[1])
                    np.testing.assert_array_equal(current[0], value + 1)
                    np.testing.assert_array_equal(current[1], value - 1)
        finally:
            model._values = bindings
        self.assertEqual(model.calls, 4)
        model.reset()
        self.assertIs(model.run(), result)

    def test_scratch_is_initialized_before_read_and_left_untouched_by_reset(self):
        class WithScratch(Block):
            x: Array = In(N)
            y: Array = Out(N)
            memory: Array = State(N, default=0)

            def setup(self, ctx):
                self.scratch = np.empty(ctx.n, dtype=ctx.dtype)

            def process(self):
                np.multiply(self.x, 2, out=self.scratch)
                np.add(self.scratch, self.memory, out=self.y)
                np.copyto(self.memory, self.y)

        model = WithScratch()
        model.setup(context())
        source = np.arange(1, 5, dtype=np.float64)
        model.x = source
        scratch, memory, output, input_view = model.scratch, model.memory, model.y, model.x
        scratch.fill(np.nan)
        result = model.run()
        np.testing.assert_array_equal(result, source * 2)
        np.testing.assert_array_equal(memory, source * 2)
        scratch.fill(123)
        model.reset()
        self.assertIs(model.scratch, scratch)
        self.assertIs(model.memory, memory)
        self.assertIs(model.y, output)
        self.assertIs(model.x, input_view)
        np.testing.assert_array_equal(scratch, 123)
        np.testing.assert_array_equal(memory, 0)
        self.assertIs(model.run(), result)
        np.testing.assert_array_equal(result, source * 2)

    def test_author_reset_initializes_custom_history_in_place(self):
        class Accumulator(Block):
            gain: float = Control(default=2)
            x: Array = In(N)
            y: Array = Out(N)
            count: int = State(default=0)

            def setup(self, ctx):
                self.history = np.empty(ctx.n, dtype=ctx.dtype)

            def reset(self):
                self.history.fill(0)

            def process(self):
                self.count += 1
                np.add(self.history, self.x, out=self.history)
                np.multiply(self.history, self.gain, out=self.y)

        model = Accumulator()
        model.setup(context())
        model.x = np.ones(4)
        history = model.history
        np.testing.assert_array_equal(history, 0)
        result = model.run()
        snapshot = result.copy()
        model.run()
        np.testing.assert_array_equal(result, 4)
        self.assertEqual(model.count, 2)
        model.reset()
        self.assertIs(model.history, history)
        np.testing.assert_array_equal(history, 0)
        np.testing.assert_array_equal(snapshot, 2)
        self.assertEqual(model.count, 0)
        self.assertEqual(model.gain, 2)
        self.assertIs(model.run(), result)
        np.testing.assert_array_equal(result, snapshot)

    def test_leaf_failure_preserves_exception_and_partial_writes_until_caller_reset(self):
        failure = FloatingPointError("author numerical failure")

        class Partial(Block):
            fail: bool = Control(default=False)
            x: Array = In(N)
            y: Array = Out(N)
            count: int = State(default=0)
            memory: Array = State(N, default=0)

            def process(self):
                self.count += 1
                self.memory += 1
                self.y[0] = self.x[0] * 2
                if self.fail:
                    raise failure
                np.multiply(self.x, 2, out=self.y)

        model = Partial()
        model.setup(context())
        source = np.arange(1, 5, dtype=np.float64)
        model.x = source
        result = model.run()
        snapshot = result.copy()
        memory = model.memory
        input_view = model.x
        model.reset()
        source[0] = 10
        model.fail = True
        with self.assertRaises(FloatingPointError) as caught:
            model.run()
        self.assertIs(caught.exception, failure)
        self.assertEqual(model.count, 1)
        np.testing.assert_array_equal(memory, 1)
        np.testing.assert_array_equal(model.y, [20, 4, 6, 8])
        with self.assertRaises(ExecutionError):
            model.run()
        np.testing.assert_array_equal(snapshot, [2, 4, 6, 8])
        self.assertIs(model.memory, memory)
        self.assertIs(model.x, input_view)
        model.reset()
        self.assertTrue(model.fail)  # The caller owns the retry policy/Control.
        self.assertEqual(model.count, 0)
        np.testing.assert_array_equal(memory, 0)
        source[0] = 1
        model.fail = False
        self.assertIs(model.run(), result)
        np.testing.assert_array_equal(result, snapshot)

    def test_composite_failure_preserves_completed_children_and_skips_later_calls(self):
        failure = np.linalg.LinAlgError("child numerical failure")

        class Stage(Block):
            fail: bool = Control(default=False)
            x: Array = In(N)
            y: Array = Out(N)
            calls: int = State(default=0)

            def process(self):
                self.calls += 1
                self.y[0] = self.x[0] * 2
                if self.fail:
                    raise failure
                np.multiply(self.x, 2, out=self.y)

        class Pipeline(Block):
            x: Array = In(N)
            y: Array = Out(N)
            started: int = State(default=0)
            finished: int = State(default=0)

            def build(self):
                self.first = self.child("first", Stage())
                self.second = self.child("second", Stage())
                self.third = self.child("third", Stage())
                self.connect(self, "x", self.first, "x")
                self.connect(self.first, "y", self.second, "x")
                self.connect(self.second, "y", self.third, "x")
                self.connect(self.third, "y", self, "y")

            def process(self):
                self.started += 1
                self.first.run()
                self.second.run()
                self.third.run()
                self.finished += 1

        model = Pipeline()
        model.setup(context())
        source = np.arange(1, 5, dtype=np.float64)
        model.x = source
        result = model.run()
        snapshot = result.copy()
        outputs = (model.first.y, model.second.y, model.third.y)
        model.reset()
        model.second.fail = True
        with self.assertRaises(np.linalg.LinAlgError) as caught:
            model.run()
        self.assertIs(caught.exception, failure)
        self.assertEqual((model.started, model.finished), (1, 0))
        self.assertEqual((model.first.calls, model.second.calls, model.third.calls), (1, 1, 0))
        np.testing.assert_array_equal(model.first.y, source * 2)
        np.testing.assert_array_equal(model.second.y, source * 4)
        model.reset()
        self.assertEqual((model.started, model.finished), (0, 0))
        self.assertEqual((model.first.calls, model.second.calls, model.third.calls), (0, 0, 0))
        for block, output in zip((model.first, model.second, model.third), outputs):
            self.assertIs(block.y, output)
        model.second.fail = False
        self.assertIs(model.run(), result)
        np.testing.assert_array_equal(result, source * 8)
        np.testing.assert_array_equal(snapshot, source * 8)
        self.assertEqual((model.started, model.finished), (1, 1))

    def test_inherited_setup_exception_unwinds_guard_and_cannot_be_reentered(self):
        events = []
        failure = ValueError("base author setup failed")

        class Base(Block):
            def setup(self, ctx):
                events.append("base")
                raise failure

        class Derived(Base):
            def setup(self, ctx):
                events.append("derived")
                super().setup(ctx)
                events.append("after super")

        model = Derived()
        with self.assertRaises(ValueError) as caught:
            model.setup(context())
        self.assertIs(caught.exception, failure)
        self.assertEqual(events, ["derived", "base"])
        for operation in (lambda: model.setup(context()), model.reset, model.run):
            with self.assertRaises(SetupError):
                operation()
        self.assertEqual(events, ["derived", "base"])
