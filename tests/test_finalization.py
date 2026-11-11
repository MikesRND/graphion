import unittest
from unittest.mock import patch

import numpy as np

from graphion import (
    Array,
    Block,
    Control,
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
from tests.test_connections import Scale, Source, context


class Relay(Block):
    target: Block = Param()
    x: Array = In(N)
    y: Array = Out(N)

    def build(self):
        self.worker = self.child("worker", self.target)
        self.connect(self, "x", self.worker, "x")
        self.connect(self.worker, "y", self, "y")

    def process(self):
        self.worker.run()


class Fanout(Block):
    x: Array = In(N)
    a_result: Array = Out(N)
    b_result: Array = Out(N)

    def build(self):
        self.a = self.child("a", Relay(target=Relay(target=Scale(gain=2.0))))
        self.b = self.child("b", Relay(target=Relay(target=Scale(gain=3.0))))
        self.connect(self, "x", self.a, "x")
        self.connect(self, "x", self.b, "x")
        self.connect(self.a, "y", self, "a_result")
        self.connect(self.b, "y", self, "b_result")

    def process(self):
        self.a.run()
        self.b.run()


def readers(model):
    return (model, model.a, model.a.worker, model.a.worker.worker, model.b, model.b.worker, model.b.worker.worker)


class FinalizationTests(unittest.TestCase):
    def test_nested_fanout_reuses_input_view_and_producer_storage(self):
        model = Fanout()
        original = np.arange(4.0)
        model.x = original
        input_view = model.x
        with (
            patch.object(arrays, "allocate", wraps=arrays.allocate) as allocate,
            patch.object(arrays, "readonly_view", wraps=arrays.readonly_view) as export,
        ):
            model.setup(context())
            self.assertEqual(allocate.call_count, 2)  # The two leaf outputs only.
            self.assertEqual(export.call_count, 2)
        for reader in readers(model):
            self.assertIs(reader.x, input_view)
            self.assertFalse(reader.x.flags.writeable)
        result = model.run()
        np.testing.assert_array_equal(result[0], original * 2)
        np.testing.assert_array_equal(result[1], original * 3)
        self.assertIs(result[0], model.a.worker.worker.run())
        self.assertIs(result[1], model.b.worker.worker.run())

    def test_missing_inputs_name_the_consumer_and_binding_clears_all_readers(self):
        model = Fanout()
        model.setup(context())
        for node, path in [(model, "root"), (model.a, "root/a"), (model.a.worker.worker, "root/a/worker/worker")]:
            with self.assertRaisesRegex(UnavailablePortError, path + r" .*unset inputs: x"):
                node.run()
        original = np.arange(4.0)
        with patch.object(arrays, "readonly_view", wraps=arrays.readonly_view) as views:
            model.x = original
            self.assertEqual(views.call_count, 1)
        for node in readers(model):
            self.assertIs(node.x, model.x)
        np.testing.assert_array_equal(model.run()[0], original * 2)
        for node in readers(model)[1:]:
            with self.assertRaises(WiringError):
                node.x = original

    def test_rebinding_updates_nested_fanout_atomically(self):
        model = Fanout()
        model.setup(context())
        original = np.arange(4.0)
        model.x = original
        previous = model.x
        model.run()
        for value, error in [
            (np.ones(3), ShapeError),
            (np.ones(4, dtype=np.float32), DtypeError),
            (model.a.worker.worker.y, StorageError),
            (None, DtypeError),
        ]:
            with self.subTest(error=error), self.assertRaises(error):
                model.x = value
            for node in readers(model):
                self.assertIs(node.x, previous)
            np.testing.assert_array_equal(model.run()[0], original * 2)
        replacement = np.full(4, 7.0)
        with patch.object(arrays, "readonly_view", side_effect=RuntimeError("view failure")):
            with self.assertRaisesRegex(RuntimeError, "view failure"):
                model.x = replacement
        for node in readers(model):
            self.assertIs(node.x, previous)
        model.x = replacement
        for node in readers(model):
            self.assertIs(node.x, model.x)
            self.assertIsNot(node.x, previous)
        np.testing.assert_array_equal(previous, original)
        np.testing.assert_array_equal(model.run()[1], replacement * 3)

    def test_run_reset_and_same_object_binding_do_not_repeat_checks_or_views(self):
        model = Fanout()
        original = np.arange(4.0)
        model.x = original
        model.setup(context())
        view, results = model.x, model.run()
        with (
            patch.object(arrays, "validate_input_storage", side_effect=AssertionError("overlap")),
            patch.object(arrays, "readonly_view", side_effect=AssertionError("view")),
            patch.object(arrays, "allocate", side_effect=AssertionError("allocation")),
            patch.object(Block, "_endpoint_source", side_effect=AssertionError("endpoint traversal")),
        ):
            model.x = original
            original.fill(4)
            self.assertIs(model.run(), results)
            model.reset()
            self.assertIs(model.run(), results)
        for node in readers(model):
            self.assertIs(node.x, view)
        np.testing.assert_array_equal(results[0], np.full(4, 8.0))
        self.assertTrue(original.flags.writeable)
        self.assertTrue(model.a.worker.worker.y.flags.writeable)

    def test_build_wiring_resolves_nested_inputs_in_either_registration_order(self):
        for reverse in (False, True):
            for source_kind in ("external", "producer"):

                class Parent(Block):
                    x: Array = In(N)

                    def build(self):
                        self.branch = Relay(target=Relay(target=Scale(gain=2.0)))
                        self.source = Source(value=7.0)
                        children = [("branch", self.branch), ("source", self.source)]
                        for name, child in reversed(children) if reverse else children:
                            self.child(name, child)
                        if source_kind == "external":
                            self.connect(self, "x", self.branch, "x")
                        else:
                            self.connect(self.source, "y", self.branch, "x")

                    def setup(self, ctx):
                        if source_kind == "external":
                            self.x = np.full(4, 5.0)

                with self.subTest(reverse=reverse, source=source_kind):
                    model = Parent()
                    model.x = np.ones(4)
                    model.setup(context())
                    if source_kind == "external":
                        for node in (model.branch, model.branch.worker, model.branch.worker.worker):
                            self.assertIs(node.x, model.x)
                        np.testing.assert_array_equal(model.branch.run(), np.full(4, 10.0))
                    else:
                        view = model.source.run()
                        for node in (model.branch, model.branch.worker, model.branch.worker.worker):
                            self.assertIs(node.x, view)
                        np.testing.assert_array_equal(model.branch.run(), np.full(4, 14.0))

    def test_ancestor_cannot_wire_nested_boundary_during_setup(self):
        class Parent(Block):
            x: Array = In(N)

            def build(self):
                self.branch = self.child("branch", Relay(target=Relay(target=Scale())))

            def setup(self, ctx):
                self.branch.setup(ctx)
                self.connect(self, "x", self.branch, "x")

        model = Parent()
        with self.assertRaisesRegex(WiringError, "before build"):
            model.setup(context())
        self.assertEqual(model.connections, ())

    def test_each_consumer_requires_only_its_own_external_sources(self):
        class Model(Block):
            active: Array = In(N)
            unused: Array = In(N)

        model = Model()
        a = model.child("a", Relay(target=Scale()))
        b = model.child("b", Relay(target=Scale()))
        model.connect(model, "active", a, "x")
        model.connect(model, "unused", b, "x")
        model.setup(context())
        model.active = np.full(4, 3.0)
        np.testing.assert_array_equal(a.run(), np.full(4, 6.0))
        with self.assertRaisesRegex(UnavailablePortError, "root/b.*unset inputs: x"):
            b.run()
        model.unused = np.full(4, 5.0)
        np.testing.assert_array_equal(b.run(), np.full(4, 10.0))
        np.testing.assert_array_equal(a.run(), np.full(4, 6.0))

    def test_multiple_consumer_ports_track_distinct_and_shared_slots(self):
        class Add(Block):
            a: Array = In(N)
            b: Array = In(N)
            y: Array = Out(N)

            def process(self):
                np.add(self.a, self.b, out=self.y)

        class Model(Block):
            x: Array = In(N)
            z: Array = In(N)

        for shared in (False, True):
            model = Model()
            child = model.child("adder", Add())
            model.connect(model, "x", child, "a")
            model.connect(model, "x" if shared else "z", child, "b")
            model.setup(context())
            model.x = np.full(4, 2.0)
            if not shared:
                with self.assertRaisesRegex(UnavailablePortError, "root/adder.*unset inputs: b"):
                    child.run()
                model.z = np.full(4, 3.0)
            np.testing.assert_array_equal(child.run(), np.full(4, 4.0 if shared else 5.0))

    def test_self_alias_through_nested_composite_boundaries_is_rejected(self):
        root = Block()
        model = root.child("model", Relay(target=Relay(target=Scale())))
        root.connect(model, "y", model, "x")
        for operation in (root.build, lambda: root.setup(context()), lambda: root.setup(context())):
            with self.assertRaisesRegex(WiringError, r"root/model/worker/worker.*\.x.*own output writer.*\.y"):
                operation()
            self.assertFalse(root._tree.failed)
            self.assertEqual(root.describe()["topology"], "open")
        for block in (root, model, model.worker, model.worker.worker):
            for operation in (block.run, block.reset):
                with self.assertRaises(SetupError):
                    operation()

    def test_nested_sibling_feedback_stops_resolution_at_producers(self):
        for reverse in (False, True):
            root = Block()
            a, b = Relay(target=Scale(gain=2.0)), Relay(target=Scale(gain=3.0))
            children = [("a", a), ("b", b)]
            for name, child in reversed(children) if reverse else children:
                root.child(name, child)
            root.connect(a, "y", b, "x")
            root.connect(b, "y", a, "x")
            root.setup(context())
            self.assertIs(a.worker.x, b.y)
            self.assertIs(b.worker.x, a.y)

    def test_root_rejects_overlapping_supplied_siblings_without_deduplication(self):
        for identical in (False, True):
            root = Block()
            a, b = root.child("a", Source()), root.child("b", Source())
            storage = np.empty(5)
            first = storage[:4]
            a.bind_output("y", first)
            b.bind_output("y", first if identical else storage[1:])
            with self.assertRaisesRegex(StorageError, "storage overlaps: root/a.*root/b"):
                root.setup(context())
            for child in (a, b):
                with self.assertRaises(SetupError):
                    child.run()

    def test_connected_consumer_output_must_not_overlap_source_storage(self):
        root = Block()
        source = root.child("source", Source())
        consumer = root.child("consumer", Relay(target=Scale()))
        storage = np.empty(4)
        source.bind_output("y", storage)
        consumer.worker.bind_output("y", storage)
        root.connect(source, "y", consumer, "x")
        with self.assertRaisesRegex(StorageError, "root/source.*root/consumer/worker"):
            root.setup(context())

    def test_external_input_prebound_before_setup_cannot_alias_descendant_output(self):
        model = Fanout()
        storage = np.ones(4)
        model.x = storage
        model.b.worker.worker.bind_output("y", storage)
        with self.assertRaisesRegex(StorageError, r"external input.*root/b/worker/worker.*\(Out\)"):
            model.setup(context())
        with self.assertRaises(SetupError):
            model.a.worker.worker.run()

    def test_late_external_input_cannot_alias_descendant_state(self):
        class Memory(Block):
            memory: Array = State(N, default=0.0)

        class Root(Block):
            x: Array = In(N)

            def build(self):
                self.branch = self.child("branch", Block())
                self.memory = self.branch.child("memory", Memory())

            def setup(self, ctx):
                self.branch.setup(ctx)
                self.x = self.memory.memory

        root = Root()
        with self.assertRaisesRegex(StorageError, r"external input.*root/branch/memory.*\(State\)"):
            root.setup(context())

    def test_supplied_output_cannot_alias_state_in_another_subtree(self):
        class Memory(Block):
            state: Array = State(N, default=0.0)

        class Root(Block):
            def build(self):
                self.memory = self.child("memory", Memory())
                self.writer = self.child("writer", Source())

            def setup(self, ctx):
                self.memory.setup(ctx)
                self.writer.bind_output("y", self.memory.state)

        with self.assertRaisesRegex(StorageError, r"root/writer.*\(Out\).*root/memory.*\(State\)"):
            Root().setup(context())

    def test_child_external_rebinding_checks_root_and_sibling_writers(self):
        class Root(Block):
            state: Array = State(N, default=0.0)

        root = Root()
        consumer = root.child("consumer", Relay(target=Scale()))
        producer = root.child("producer", Source())
        root.setup(context())
        original = np.arange(4.0)
        consumer.x = original
        view = consumer.x
        for candidate in (root.state, producer.y, consumer.worker.y):
            with self.assertRaises(StorageError):
                consumer.x = candidate
            self.assertIs(consumer.x, view)
            self.assertIs(consumer.worker.x, view)
        np.testing.assert_array_equal(consumer.run(), original * 2)

    def test_interleaved_tree_storage_and_bounded_overlap_checks(self):
        model = Fanout()
        bank = np.empty(12)
        model.a.worker.worker.bind_output("y", bank[::3])
        model.b.worker.worker.bind_output("y", bank[1::3])
        source = bank[2::3]
        source.fill(5)
        model.x = source
        model.setup(context())
        result = model.run()
        np.testing.assert_array_equal(result[0], np.full(4, 10.0))
        np.testing.assert_array_equal(result[1], np.full(4, 15.0))
        view = model.x
        candidate = source.view()  # New identity, with disjoint but interleaved bytes.
        with patch.object(np, "shares_memory", side_effect=np.exceptions.TooHardError) as exact:
            with self.assertRaisesRegex(StorageError, "work limit.*external input.*root/a"):
                model.x = candidate
            self.assertGreater(exact.call_args.kwargs["max_work"], 0)
        for node in readers(model):
            self.assertIs(node.x, view)
        model.x = candidate
        for node in readers(model):
            self.assertIs(node.x, model.x)
        with self.assertRaisesRegex(StorageError, "storage overlaps"):
            model.x = bank[::3]

    def test_failed_first_bind_keeps_all_forwarded_consumers_unready(self):
        model = Fanout()
        model.setup(context())
        with self.assertRaises(StorageError):
            model.x = model.b.worker.worker.y
        for node in readers(model):
            with self.assertRaises(UnavailablePortError):
                _ = node.x
            with self.assertRaises(UnavailablePortError):
                node.run()

    def test_scalar_control_forwarding_needs_only_explicit_execution(self):
        class Controlled(Relay):
            gain: float = Control(default=1.0)

            def process(self):
                self.worker.gain = self.gain
                self.worker.run()

        model = Controlled(target=Scale())
        model.setup(context())
        model.x = np.arange(4.0)
        result = model.run()
        np.testing.assert_array_equal(result, np.arange(4.0))
        model.gain = 3.0
        self.assertIs(model.run(), result)
        self.assertEqual(model.worker.gain, 3.0)
        np.testing.assert_array_equal(result, np.arange(4.0) * 3)
        model.reset()
        self.assertEqual(model.gain, 3.0)
        self.assertIs(model.run(), result)
