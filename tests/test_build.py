"""Context-free topology validation and the public build boundary."""

import unittest
from unittest.mock import patch

import numpy as np

from graphion import (
    Array,
    Axis,
    Block,
    C,
    DtypeError,
    In,
    N,
    Out,
    RegistrationError,
    SetupError,
    ShapeError,
    State,
    UnavailablePortError,
    WiringError,
)
from graphion.dsp.blocks import Gain
from graphion.profile.numpy_frame import _storage as arrays
from tests.test_connections import Boundary, Scale, Source, context
from tests.test_describe import declarations
from tests.test_finalization import Relay


class BuildTests(unittest.TestCase):
    def test_constructor_runs_most_derived_author_body_once_with_super(self):
        events = []

        class Base(Block):
            def build(self):
                super().build()
                events.append("base")
                self.first = self.child("first", Block())

        class Derived(Base):
            def build(self):
                events.append("derived before")
                super().build()
                self.last = self.child("last", Block())
                events.append("derived after")

        model = Derived()
        self.assertEqual(events, ["derived before", "base", "derived after"])
        self.assertEqual(dict(model.children), {"first": model.first, "last": model.last})
        self.assertEqual(model.describe()["topology"], "open")
        model.build()
        model.build()
        model.setup(context())
        model.build()
        self.assertEqual(events, ["derived before", "base", "derived after"])

    def test_root_build_certifies_topology_without_resolving_interfaces(self):
        model = Gain()
        child = model.child("child", Gain())
        model.connect(model, "x", child, "x")
        model.build()
        snapshot = model.describe()
        self.assertEqual(snapshot["topology"], "complete")
        self.assertEqual(snapshot["schema_version"], 2)
        for record in snapshot["blocks"]:
            self.assertIsNone(record["context"])
            self.assertIsNone(record["storage"])
            self.assertEqual(
                declarations(record)["x"]["shape"],
                [{"kind": "symbol", "name": "C"}, {"kind": "symbol", "name": "N"}],
            )
        self.assertNotIn("topology", [fact["fact"] for fact in snapshot["unresolved"]])
        with patch.object(Block, "_endpoint_source", side_effect=AssertionError("repeat validation")):
            self.assertIsNone(model.build())
        self.assertEqual(model.describe(), snapshot)

    def test_build_does_not_prepare_storage_evaluate_axes_or_run_hooks(self):
        class Model(Block):
            x: Array = In(C, N)
            y: Array = Out(Axis("width"))
            memory: Array = State(N, default=0.0)

            @property
            def width(self):
                raise AssertionError("axis was evaluated")

            def setup(self, ctx):
                raise AssertionError("setup was called")

            def reset(self):
                raise AssertionError("reset was called")

            def process(self):
                raise AssertionError("process was called")

        model = Model()
        model.child("child", Model())
        with patch.object(arrays, "allocate", side_effect=AssertionError("allocation")):
            model.build()
        for node in model._walk():
            self.assertEqual(node._specs, {})
            self.assertEqual(node._values, {})
            self.assertIsNone(node._ctx)
        self.assertEqual(model.describe()["topology"], "complete")

    def test_interface_mismatches_are_deferred_to_setup(self):
        for shape, dtype, error in [(3, np.float64, ShapeError), (4, np.float32, DtypeError)]:
            with self.subTest(error=error):

                class Consumer(Block):
                    x: Array = In(shape, dtype=dtype)

                model = Block()
                source = model.child("source", Source())
                consumer = model.child("consumer", Consumer())
                model.connect(source, "y", consumer, "x")
                model.build()
                self.assertEqual(model.describe()["topology"], "complete")
                with self.assertRaises(error):
                    model.setup(context())
                self.assertEqual(model.describe()["topology"], "complete")
                self.assertTrue(all(record["storage"] is None for record in model.describe()["blocks"]))

    def test_built_tree_is_not_runnable_or_resettable(self):
        model = Gain()
        child = model.child("child", Gain())
        model.build()
        for node in (model, child):
            for operation in (node.run, node.run_output, node.run_update, node.reset):
                with self.subTest(operation=operation), self.assertRaises(SetupError):
                    operation()

    def test_build_after_setup_preserves_phase_buffers_and_bindings(self):
        model = Relay(target=Scale())
        model.x = np.arange(4.0)
        model.setup(context())
        snapshot = model.describe()
        output, direct, bound = model.y, model.worker.y, model.x
        with patch.object(Block, "_endpoint_source", side_effect=AssertionError("repeat validation")):
            self.assertIsNone(model.build())
        self.assertEqual(model._phase, "configured")
        self.assertEqual(model.worker._phase, "configured")
        self.assertIs(model.y, output)
        self.assertIs(model.worker.y, direct)
        self.assertIs(model.x, bound)
        self.assertIs(model.worker.x, bound)
        self.assertEqual(model.describe(), snapshot)
        np.testing.assert_array_equal(model.run(), np.arange(4.0) * 2)

    def test_registered_child_cannot_build_at_any_stage(self):
        model = Block()
        child = model.child("child", Gain())
        for prepare in (lambda: None, model.build, lambda: model.setup(context())):
            prepare()
            before = model.describe()
            phases = (model._phase, child._phase)
            with self.assertRaisesRegex(SetupError, "lifecycle root"):
                child.build()
            self.assertEqual(model.describe(), before)
            self.assertEqual((model._phase, child._phase), phases)

    def test_build_freezes_every_owner_but_allows_input_and_output_binding(self):
        class Node(Boundary):
            count: int = State(default=0)

        model = Node()
        branch = model.child("branch", Node())
        leaf = branch.child("leaf", Node())
        model.build()
        for node in (model, branch, leaf):
            with self.assertRaisesRegex(RegistrationError, "before build"):
                node.child("late", Block())
            with self.assertRaisesRegex(WiringError, "before build"):
                node.connect(node, "x", node, "x")
            with self.assertRaises(UnavailablePortError):
                node.count = 1
            node.x = np.arange(4.0)
            node.bind_output("y", np.empty(4))
        outputs = [node._supplied["y"] for node in (model, branch, leaf)]
        model.setup(context())
        for node, output in zip((model, branch, leaf), outputs):
            self.assertIs(node.y, output)
            self.assertEqual(node.count, 0)
            np.testing.assert_array_equal(node.x, np.arange(4.0))

    def test_failed_build_preserves_frozen_subtree_and_editable_owners(self):
        subtree = Relay(target=Relay(target=Scale()))
        subtree.build()
        model = Boundary()
        sibling = model.child("sibling", Boundary())
        model.child("subtree", subtree)
        model.connect(subtree, "y", subtree, "x")
        before = model.describe()
        connections = model.connections
        for operation in (model.build, lambda: model.setup(context())):
            with self.assertRaisesRegex(WiringError, "own output writer"):
                operation()
            self.assertEqual(model.describe(), before)
            self.assertIs(model.connections, connections)
            self.assertFalse(model._tree.failed)
            self.assertEqual((model._phase, sibling._phase), ("new", "new"))
            for node in subtree._walk():
                self.assertEqual(node._phase, "built")
                with self.assertRaisesRegex(RegistrationError, "before build"):
                    node.child("late", Block())
                with self.assertRaisesRegex(WiringError, "before build"):
                    node.connect(node, "x", node, "x")
        model.child("extra", Block())
        sibling.child("extra", Block())
        model.connect(model, "x", sibling, "x")
        source = sibling.child("source", Source())
        consumer = sibling.child("consumer", Scale())
        sibling.connect(source, "y", consumer, "x")
        self.assertEqual(model.describe()["topology"], "open")

    def test_setup_automatically_builds_and_later_failure_preserves_complete_topology(self):
        model = Gain()
        model.setup(context())
        self.assertEqual(model.describe()["topology"], "complete")
        self.assertIsNotNone(model.describe()["blocks"][0]["storage"])

        class Failing(Block):
            def setup(self, ctx):
                raise RuntimeError("author setup failed")

        for model, ctx, error in [(Failing(), context(), RuntimeError), (Block(), None, SetupError)]:
            with self.subTest(error=error), self.assertRaises(error):
                model.setup(ctx)
            snapshot = model.describe()
            self.assertEqual(snapshot["topology"], "complete")
            self.assertIsNone(snapshot["blocks"][0]["storage"])
            self.assertTrue(model._tree.failed)
            self.assertIsNone(model.build())
            self.assertEqual(model.describe(), snapshot)
            with self.assertRaises(SetupError):
                model.setup(context())
            with self.assertRaises(SetupError):
                model.run()

    def test_build_rejection_via_setup_preserves_editability(self):
        model = Block()
        worker = model.child("worker", Scale())
        model.connect(worker, "y", worker, "x")
        before = model.describe()
        with self.assertRaises(WiringError):
            model.setup(context())
        self.assertEqual(model.describe(), before)
        self.assertFalse(model._tree.failed)
        model.child("extra", Block())
        worker.child("extra", Block())
        self.assertEqual(model.describe()["topology"], "open")

    def test_built_subtree_adoption_uses_current_root_and_resolves_new_boundary(self):
        subtree = Relay(target=Relay(target=Scale()))
        subtree.build()
        self.assertEqual(subtree.describe()["topology"], "complete")
        parent = Boundary()
        self.assertIs(parent.child("s", subtree), subtree)
        self.assertEqual(subtree.describe()["topology"], "open")
        with self.assertRaises(RegistrationError):
            subtree.child("late", Block())
        parent.connect(parent, "x", subtree, "x")
        parent.connect(subtree, "y", parent, "y")
        parent.build()
        self.assertEqual(subtree.describe()["topology"], "complete")
        parent.setup(context())
        parent.x = np.arange(4.0)
        for node in subtree._walk():
            self.assertIs(node.x, parent.x)
        np.testing.assert_array_equal(subtree.run(), np.arange(4.0) * 2)
        replacement = np.full(4, 3.0)
        parent.x = replacement
        for node in subtree._walk():
            self.assertIs(node.x, parent.x)
        np.testing.assert_array_equal(subtree.run(), replacement * 2)

    def test_explicit_child_setup_accepts_built_phase_and_derived_context(self):
        class Parent(Block):
            def build(self):
                self.worker = self.child("worker", Gain())

            def setup(self, ctx):
                self.worker.setup(ctx.replace(n=2, rate=4))

        parent = Parent()
        parent.setup(context())
        self.assertEqual((parent.worker.ctx.n, parent.worker.ctx.rate), (2, 4))
        self.assertEqual(parent.worker.y.shape, (1, 2))
        parent.worker.x = np.ones((1, 2))
        np.testing.assert_array_equal(parent.worker.run(), np.ones((1, 2)))
