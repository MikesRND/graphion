"""Caller-ordered schedules through registered public-API connections."""

import unittest

import numpy as np

from graphion import Array, Block, C, Control, Ctx, In, N, Out, State
from graphion.dsp.blocks import FrameDelay, Gain
from tests.test_documented_examples import module_with, reference

DTYPES = (np.float32, np.float64, np.complex64, np.complex128)


def context(dtype):
    return Ctx(rate=8, channels=2, n=4, dtype=dtype)


def signal(dtype):
    result = np.arange(1, 9).reshape(2, 4).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        result += 1j * np.arange(8, 0, -1).reshape(2, 4)
    return result


class DelaySeries(Block):
    x: Array = In(C, N)
    y: Array = Out(C, N)

    def build(self):
        self.first = self.child("first", FrameDelay())
        self.second = self.child("second", FrameDelay())
        self.connect(self, "x", self.first, "x")
        self.connect(self.first, "y", self.second, "x")
        self.connect(self.second, "y", self, "y")

    def process(self):
        self.first.run()
        self.second.run()


class CallCounter(Block):
    """Every invocation increments calls once and emits input plus calls."""

    calls: int = State(default=0)
    x: Array = In(C, N)
    y: Array = Out(C, N)

    def process(self):
        self.calls += 1
        np.add(self.x, self.calls, out=self.y)


class ConditionalCalls(Block):
    enabled: bool = Control(default=True)
    repetitions: int = Control(default=1)
    x: Array = In(C, N)
    y: Array = Out(C, N)

    def build(self):
        self.worker = self.child("worker", CallCounter())
        self.connect(self, "x", self.worker, "x")
        self.connect(self.worker, "y", self, "y")

    def process(self):
        if self.enabled:
            for _ in range(self.repetitions):
                self.worker.run()


class GenerationFanout(Block):
    x: Array = In(C, N)
    before: Array = Out(C, N)
    after: Array = Out(C, N)

    def build(self):
        self.producer = self.child("producer", CallCounter())
        self.early = self.child("early", Gain())
        self.late = self.child("late", Gain())
        self.connect(self, "x", self.producer, "x")
        self.connect(self.producer, "y", self.early, "x")
        self.connect(self.producer, "y", self.late, "x")
        self.connect(self.early, "y", self, "before")
        self.connect(self.late, "y", self, "after")

    def process(self):
        self.producer.run()
        self.early.run()
        self.producer.run()
        self.late.run()


class ScheduleTests(unittest.TestCase):
    def test_frame_delays_in_series_preserve_dtype_and_two_call_history_after_reset(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                model = DelaySeries()
                model.setup(context(dtype))
                base = signal(dtype)
                source = np.empty_like(base)
                model.x = source
                buffers = (model.first.held, model.second.held, model.first.y, model.second.y)
                result = None
                for replay in range(2):
                    for index in range(5):
                        source[:] = base * (index + 1)
                        actual = model.run()
                        if result is None:
                            result = actual
                        self.assertIs(actual, result)
                        self.assertEqual(actual.dtype, np.dtype(dtype))
                        expected = np.zeros_like(base) if index < 2 else base * (index - 1)
                        np.testing.assert_array_equal(actual, expected)
                    model.reset()
                    for before, after in zip(
                        buffers, (model.first.held, model.second.held, model.first.y, model.second.y)
                    ):
                        self.assertIs(before, after)
                    np.testing.assert_array_equal(model.first.held, 0)
                    np.testing.assert_array_equal(model.second.held, 0)

    def test_documented_feedback_recurrence_and_reset_for_real_and_complex_frames(self):
        with module_with(reference("### Feedback with explicit State")) as example:
            for dtype in DTYPES:
                with self.subTest(dtype=dtype):
                    model = example.Feedback()
                    model.setup(context(dtype))
                    impulse = signal(dtype)
                    source = np.empty_like(impulse)
                    model.x = source
                    history = model.previous
                    self.assertFalse(np.shares_memory(model.y, history))
                    for replay in range(2):
                        for index, weight in enumerate((1, 0.5, 0.25, 0.125, 0.0625)):
                            source[:] = impulse if index == 0 else 0
                            np.testing.assert_array_equal(model.run(), impulse * weight)
                        model.reset()
                        self.assertIs(model.previous, history)
                        np.testing.assert_array_equal(history, 0)

    def test_repeated_and_conditionally_skipped_runs_update_state_once_per_invocation(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                model = ConditionalCalls()
                model.setup(context(dtype))
                source = signal(dtype)
                model.x = source
                input_view = model.worker.x
                result = model.run()
                self.assertEqual(model.worker.calls, 1)
                np.testing.assert_array_equal(result, source + 1)
                self.assertIs(model.run(), result)  # No rebind or new input frame.
                self.assertEqual(model.worker.calls, 2)
                np.testing.assert_array_equal(result, source + 2)
                model.enabled = False
                snapshot = result.copy()
                self.assertIs(model.run(), result)
                self.assertEqual(model.worker.calls, 2)
                # The Direct Producer did not run, so its existing Borrow stays valid.
                np.testing.assert_array_equal(result, snapshot)
                model.enabled = True
                model.repetitions = 2
                self.assertIs(model.run(), result)
                self.assertEqual(model.worker.calls, 4)
                np.testing.assert_array_equal(result, source + 4)
                self.assertIs(model.worker.x, input_view)
                model.reset()
                self.assertEqual(model.worker.calls, 0)
                np.testing.assert_array_equal(model.run(), source + 2)
                self.assertEqual(model.worker.calls, 2)

    def test_fanout_consumers_read_different_generations_of_the_same_producer_buffer(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                model = GenerationFanout()
                model.setup(context(dtype))
                source = signal(dtype)
                model.x = source
                self.assertIs(model.early.x, model.late.x)
                self.assertTrue(np.shares_memory(model.early.x, model.producer.y))
                self.assertFalse(model.early.x.flags.writeable)
                result = None
                for call in range(1, 4):
                    actual = model.run()
                    if result is None:
                        result = actual
                    self.assertIs(actual, result)
                    before, after = actual
                    np.testing.assert_array_equal(before, source + 2 * call - 1)
                    np.testing.assert_array_equal(after, source + 2 * call)
                    self.assertEqual(model.producer.calls, 2 * call)
                    self.assertIs(model.early.x, model.late.x)
                model.reset()
                before, after = model.run()
                np.testing.assert_array_equal(before, source + 1)
                np.testing.assert_array_equal(after, source + 2)
                self.assertEqual(model.producer.calls, 2)
