import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np

from graphion import (
    Array,
    Block,
    BlockError,
    Control,
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
    UnavailablePortError,
    WiringError,
)
from graphion.profile.numpy_frame import _storage as arrays


def context():
    return Ctx(rate=8, n=4, channels=1)


class Source(Block):
    value: float = Control(default=1.0)
    calls: int = State(default=0)
    y: Array = Out(N)

    def process(self):
        self.calls += 1
        self.y.fill(self.value)


class Scale(Block):
    gain: float = Control(default=2.0)
    x: Array = In(N)
    y: Array = Out(N)

    def process(self):
        np.multiply(self.x, self.gain, out=self.y)


class Boundary(Block):
    x: Array = In(N)
    y: Array = Out(N)


class Forward(Block):
    supplied: Block = Param()
    y: Array = Out(N)

    def build(self):
        self.producer = self.child("producer", self.supplied)
        self.connect(self.producer, "y", self, "y")

    def process(self):
        self.producer.run()


class ConnectionTests(unittest.TestCase):
    def test_three_patterns_have_immutable_ordered_records(self):
        parent = Boundary()
        a = parent.child("a", Scale())
        b = parent.child("b", Scale())
        endpoints = [(parent, "x", a, "x"), (a, "y", b, "x"), (b, "y", parent, "y")]
        for edge in endpoints:
            parent.connect(*edge)
        records = parent.connections
        self.assertIsInstance(records, tuple)
        for record, (source, source_port, destination, destination_port) in zip(records, endpoints):
            self.assertIs(record.source_block, source)
            self.assertIs(record.destination_block, destination)
            self.assertEqual(record.source_port, source_port)
            self.assertEqual(record.destination_port, destination_port)
            with self.assertRaises(FrozenInstanceError):
                record.source_port = "other"
        with self.assertRaises(AttributeError):
            parent.connections = ()
        with self.assertRaises(TypeError):
            records[0] = records[1]
        parent.setup(context())
        self.assertIs(parent.connections, records)
        self.assertTrue(np.shares_memory(parent.y, b.y))
        self.assertFalse(parent.y.flags.writeable)

    def test_invalid_endpoints_and_patterns_leave_records_unchanged(self):
        parent = Boundary()
        child = parent.child("child", Scale())
        grandchild = child.child("grandchild", Scale())
        outsider = Scale()
        cases = [
            (outsider, "y", child, "x"),
            (grandchild, "y", child, "x"),
            (child, "y", outsider, "x"),
            (object(), "y", child, "x"),
            (child, "gain", parent, "y"),
            (parent, "missing", child, "x"),
            (parent, [], child, "x"),
            (child, "y", parent, None),
            (parent, "x", parent, "y"),
            (parent, "y", child, "x"),
            (child, "x", parent, "y"),
            (child, "y", parent, "x"),
            (parent, "x", child, "y"),
            (child, "y", child, "y"),
        ]
        for edge in cases:
            with self.subTest(edge=edge), self.assertRaises(WiringError) as caught:
                parent.connect(*edge)
            self.assertIsInstance(caught.exception, BlockError)
            self.assertEqual(parent.connections, ())
        # Rejected records did not reserve either valid destination.
        parent.connect(parent, "x", child, "x")
        parent.connect(child, "y", parent, "y")
        self.assertEqual(len(parent.connections), 2)

    def test_duplicate_sources_are_rejected_and_fanout_is_shared(self):
        parent = Block()
        source = parent.child("source", Source(value=3.0))
        a = parent.child("a", Scale(gain=2.0))
        b = parent.child("b", Scale(gain=4.0))
        parent.connect(source, "y", a, "x")
        original = parent.connections
        for producer in (source, b):
            with self.assertRaises(WiringError):
                parent.connect(producer, "y", a, "x")
            self.assertIs(parent.connections, original)
        parent.connect(source, "y", b, "x")
        parent.setup(context())
        export = source.run()
        self.assertIs(a.x, export)
        self.assertIs(b.x, export)
        self.assertFalse(export.flags.writeable)
        self.assertTrue(source.y.flags.writeable)
        np.testing.assert_array_equal(a.run(), np.full(4, 6.0))
        np.testing.assert_array_equal(b.run(), np.full(4, 12.0))
        self.assertEqual(source.calls, 1)  # Reading a source never schedules it.

    def test_sibling_connections_work_in_either_registration_order(self):
        for reverse in (False, True):
            parent = Block()
            source, consumer = Source(value=5.0), Scale(gain=3.0)
            children = [("source", source), ("consumer", consumer)]
            for name, child in reversed(children) if reverse else children:
                parent.child(name, child)
            parent.connect(source, "y", consumer, "x")
            parent.setup(context())
            # Setup allocates storage without running the producer.
            self.assertEqual(source.calls, 0)
            output = source.run()
            self.assertIs(consumer.x, output)
            np.testing.assert_array_equal(consumer.run(), np.full(4, 15.0))

    def test_parent_can_wire_explicitly_configured_children(self):
        class Parent(Block):
            def build(self):
                self.consumer = self.child("consumer", Scale(gain=3.0))
                self.source = self.child("source", Source(value=2.0))

            def setup(self, ctx):
                self.consumer.setup(ctx)
                self.source.setup(ctx)
                self.connect(self.source, "y", self.consumer, "x")

        parent = Parent()
        parent.setup(context())
        export = parent.source.run()
        self.assertIs(parent.consumer.x, export)
        np.testing.assert_array_equal(parent.consumer.run(), np.full(4, 6.0))

    def test_external_input_conflict_preserves_original_binding(self):
        for configured in (False, True):
            source_array = np.arange(4.0)
            test = self

            class Parent(Block):
                def build(self):
                    self.source = self.child("source", Source())
                    self.consumer = self.child("consumer", Scale())
                    self.consumer.x = source_array

                def setup(self, ctx):
                    if configured:
                        self.consumer.setup(ctx)
                    before = self.consumer.x
                    with test.assertRaises(WiringError):
                        self.connect(self.source, "y", self.consumer, "x")
                    test.assertIs(self.consumer.x, before)

            parent = Parent()
            parent.setup(context())
            self.assertEqual(parent.connections, ())
            np.testing.assert_array_equal(parent.consumer.run(), source_array * 2)

    def test_connected_input_assignment_is_rejected_before_and_after_setup(self):
        parent = Boundary()
        source = parent.child("source", Source())
        fixed = parent.child("fixed", Scale())
        forwarded = parent.child("forwarded", Scale())
        parent.connect(source, "y", fixed, "x")
        parent.connect(parent, "x", forwarded, "x")
        for child in (fixed, forwarded):
            with self.assertRaises(WiringError):
                child.x = np.zeros(4)
        parent.setup(context())
        original = fixed.x
        for value in (original, np.ones(4), None):
            with self.assertRaises(WiringError):
                fixed.x = value
            self.assertIs(fixed.x, original)
        with self.assertRaises(WiringError):
            forwarded.x = np.ones(4)

    def test_late_or_reset_time_wiring_is_rejected(self):
        test = self

        class Parent(Block):
            def build(self):
                self.source = self.child("source", Source())
                self.consumer = self.child("consumer", Scale())

            def reset(self):
                with test.assertRaises(WiringError):
                    self.connect(self.source, "y", self.consumer, "x")

        parent = Parent()
        parent.setup(context())
        parent.reset()
        with self.assertRaises(WiringError):
            parent.connect(parent.source, "y", parent.consumer, "x")
        self.assertEqual(parent.connections, ())
        with self.assertRaises(UnavailablePortError):
            parent.consumer.run()

        failed = Parent()
        with self.assertRaises(SetupError):
            failed.setup(None)
        with self.assertRaises(WiringError):
            failed.connect(failed.source, "y", failed.consumer, "x")

    def test_all_connection_patterns_validate_shape_and_dtype(self):
        def node(length, dtype):
            class Node(Block):
                x: Array = In(length, dtype=dtype)
                y: Array = Out(length, dtype=dtype)

            return Node()

        for length, dtype, error in [(3, np.float64, ShapeError), (2, np.float32, DtypeError)]:
            for pattern in ("input", "sibling", "output"):
                with self.subTest(pattern=pattern, error=error):
                    parent = node(2, np.float64)
                    source = parent.child("source", node(2, np.float64))
                    different = parent.child("different", node(length, dtype))
                    if pattern == "input":
                        parent.connect(parent, "x", different, "x")
                    elif pattern == "sibling":
                        parent.connect(source, "y", different, "x")
                    else:
                        parent.connect(different, "y", parent, "y")
                    with self.assertRaisesRegex(error, "connection .* differ"):
                        parent.setup(context())
                    for child in (parent, source, different):
                        with self.assertRaises(SetupError):
                            child.run()

    def test_nested_forwarding_reuses_one_buffer_and_one_export(self):
        source = Source(value=3.0)
        inner = Forward(supplied=source)
        outer = Forward(supplied=inner)
        with self.assertRaises(UnavailablePortError):
            _ = outer.y
        with (
            patch.object(arrays, "allocate", wraps=arrays.allocate) as allocate,
            patch.object(arrays, "readonly_view", wraps=arrays.readonly_view) as export,
        ):
            outer.setup(context())
            self.assertEqual(allocate.call_count, 1)
            self.assertEqual(export.call_count, 1)
        view = outer.run()
        self.assertIs(view, outer.y)
        self.assertIs(view, inner.y)
        self.assertIs(view, source.run())
        self.assertIsNot(view, source.y)
        self.assertTrue(np.shares_memory(view, source.y))
        self.assertFalse(view.flags.writeable)
        with self.assertRaises(ValueError):
            outer.y[:] = 7
        with (
            patch.object(arrays, "allocate", side_effect=AssertionError("run allocation")),
            patch.object(arrays, "readonly_view", side_effect=AssertionError("run export")),
        ):
            self.assertIs(outer.run(), view)
            outer.reset()
            self.assertIs(outer.y, view)
            self.assertIs(outer.run(), view)

    def test_forwarded_result_changes_when_direct_producer_runs(self):
        source = Source(value=2.0)
        parent = Forward(supplied=Forward(supplied=source))
        parent.setup(context())
        view = parent.run()
        saved = view.copy()
        source.value = 7.0
        self.assertIs(source.run(), view)
        np.testing.assert_array_equal(view, np.full(4, 7.0))
        np.testing.assert_array_equal(saved, np.full(4, 2.0))
        parent.reset()
        self.assertIs(parent.y, view)
        np.testing.assert_array_equal(parent.run(), np.full(4, 7.0))

    def test_sibling_source_can_be_a_nested_forwarded_output(self):
        root = Block()
        source = Source(value=5.0)
        consumer = root.child("consumer", Scale(gain=3.0))
        branch = root.child("branch", Forward(supplied=Forward(supplied=source)))
        root.connect(branch, "y", consumer, "x")
        root.setup(context())
        view = source.run()
        self.assertIs(branch.y, view)
        self.assertIs(consumer.x, view)
        np.testing.assert_array_equal(consumer.run(), np.full(4, 15.0))

    def test_mixed_outputs_return_cached_tuple_in_declaration_order(self):
        class Parent(Block):
            first: Array = Out(N)
            direct: Array = Out(N)
            last: Array = Out(N)

            def build(self):
                self.source = self.child("source", Source(value=2.0))
                self.connect(self.source, "y", self, "last")
                self.connect(self.source, "y", self, "first")

            def process(self):
                self.source.run()
                self.direct.fill(7)

        parent = Parent()
        parent.setup(context())
        result = parent.run()
        self.assertIs(result[0], parent.first)
        self.assertIs(result[2], parent.last)
        self.assertIs(result[0], result[2])
        self.assertIsNot(result[1], parent.direct)
        self.assertTrue(np.shares_memory(result[1], parent.direct))
        self.assertFalse(result[1].flags.writeable)
        np.testing.assert_array_equal(result[0], np.full(4, 2.0))
        np.testing.assert_array_equal(result[1], np.full(4, 7.0))
        self.assertIs(parent.run(), result)
        parent.reset()
        self.assertIs(parent.run(), result)

    def test_binding_forwarded_output_conflicts_are_atomic_in_both_orders(self):
        buffer = np.zeros(4)
        parent = Boundary()
        source = parent.child("source", Source())
        parent.connect(source, "y", parent, "y")
        records = parent.connections
        with self.assertRaises(StorageError):
            parent.bind_output("y", buffer)
        self.assertIs(parent.connections, records)
        source.bind_output("y", buffer)
        parent.setup(context())
        self.assertIs(source.y, buffer)
        self.assertIs(parent.y, source.run())

        other = Boundary()
        producer = other.child("source", Source())
        other.bind_output("y", buffer)
        records = other.connections
        with self.assertRaises(StorageError):
            other.connect(producer, "y", other, "y")
        self.assertIs(other.connections, records)
        other.setup(context())
        self.assertIs(other.y, buffer)
        self.assertTrue(other.y.flags.writeable)

    def test_sibling_cycle_with_distinct_writers_resolves(self):
        for reverse in (False, True):
            parent = Block()
            a, b = Scale(gain=2.0), Scale(gain=3.0)
            nodes = [("a", a), ("b", b)]
            for name, block in reversed(nodes) if reverse else nodes:
                parent.child(name, block)
            parent.connect(a, "y", b, "x")
            parent.connect(b, "y", a, "x")
            parent.setup(context())
            # Endpoint resolution permits cycles without claiming a valid Schedule.
            self.assertTrue(np.shares_memory(a.x, b.y))
            self.assertTrue(np.shares_memory(b.x, a.y))

    def test_direct_self_alias_is_rejected_during_resolution(self):
        parent = Block()
        worker = parent.child("worker", Scale())
        parent.connect(worker, "y", worker, "x")
        self.assertEqual(len(parent.connections), 1)
        with self.assertRaisesRegex(StorageError, "own output writer"):
            parent.setup(context())
        with self.assertRaises(SetupError):
            worker.run()
