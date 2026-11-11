import unittest

import numpy as np

from graphion import (
    Array,
    Axis,
    Block,
    BlockError,
    Control,
    Ctx,
    N,
    Out,
    Param,
    RegistrationError,
    SetupError,
    State,
    UnavailablePortError,
    WiringError,
)
from graphion.dsp.blocks import Gain


def context():
    return Ctx(rate=8, n=4, channels=1)


class HierarchyTests(unittest.TestCase):
    def test_registration_returns_child_and_exposes_ordered_readonly_view(self):
        class Parent(Block):
            supplied: Block = Param()

            def build(self):
                self.first = self.child("first /~", self.supplied)
                self.last = self.child("last", Block())

        supplied = Block()
        parent = Parent(supplied=supplied)
        view = parent.children
        self.assertIs(parent.first, supplied)
        self.assertIs(view["first /~"], supplied)
        with self.assertRaises(TypeError):
            view["other"] = Block()
        with self.assertRaises(AttributeError):
            parent.children = {}
        parent.setup(context())
        self.assertIs(parent.children, view)
        self.assertEqual(list(view), ["first /~", "last"])
        self.assertIs(parent.last.ctx, parent.ctx)

    def test_registration_in_setup_body_is_rejected(self):
        class Parent(Block):
            def setup(self, ctx):
                self.child("late", Block())

        parent = Parent()
        with self.assertRaisesRegex(RegistrationError, "before build"):
            parent.setup(context())
        self.assertEqual(dict(parent.children), {})

    def test_invalid_registration_is_atomic(self):
        class Parent(Block):
            reserved: int = Param(default=1)

        parent = Parent()
        existing = parent.child("existing", Block())
        for name, child in [
            ("", Block()),
            (None, Block()),
            (3, Block()),
            ("reserved", Block()),
            ("existing", Block()),
            ("invalid", object()),
            ("self", parent),
            ("again", existing),
        ]:
            with self.subTest(name=name), self.assertRaises(RegistrationError) as caught:
                parent.child(name, child)
            self.assertIsInstance(caught.exception, BlockError)
            self.assertEqual(dict(parent.children), {"existing": existing})
        other = Block()
        with self.assertRaises(RegistrationError):
            other.child("stolen", existing)
        self.assertEqual(dict(other.children), {})
        self.assertIs(parent.children["existing"], existing)
        # A rejected name did not attach the otherwise-valid candidate.
        candidate = Block()
        with self.assertRaises(RegistrationError):
            parent.child("reserved", candidate)
        self.assertIs(other.child("valid", candidate), candidate)

    def test_cycles_configured_children_and_late_registration_are_rejected(self):
        root = Block()
        middle = root.child("middle", Block())
        leaf = middle.child("leaf", Block())
        with self.assertRaises(RegistrationError):
            leaf.child("cycle", root)
        self.assertEqual(dict(leaf.children), {})
        configured = Block()
        configured.setup(context())
        with self.assertRaises(RegistrationError):
            root.child("configured", configured)
        failed = Block()
        with self.assertRaises(SetupError):
            failed.setup(None)
        with self.assertRaises(RegistrationError):
            root.child("failed", failed)
        root.setup(context())
        for node in (root, middle, leaf, failed):
            with self.subTest(node=node), self.assertRaises(RegistrationError):
                node.child("late", Block())

    def test_recursive_setup_and_reset_order_with_explicit_context(self):
        events = []

        class Leaf(Block):
            label: str = Param()
            count: int = State(default=0)

            def setup(self, ctx):
                events.append((self.label, "setup", ctx.n))

            def reset(self):
                self.count += 1
                events.append((self.label, "reset", self.count))

        class Parent(Block):
            count: int = State(default=0)

            def build(self):
                self.first = self.child("first", Leaf(label="first"))
                self.second = self.child("second", Leaf(label="second"))

            def setup(self, ctx):
                events.append(("parent", "setup", ctx.n))
                self.second.setup(ctx.replace(n=2, rate=4))

            def reset(self):
                self.count += 1
                events.append(("parent", "reset", self.first.count, self.second.count))

        parent = Parent()
        parent.setup(context())
        self.assertEqual(
            events,
            [
                ("parent", "setup", 4),
                ("second", "setup", 2),
                ("second", "reset", 1),
                ("first", "setup", 4),
                ("first", "reset", 1),
                ("parent", "reset", 1, 1),
            ],
        )
        self.assertIs(parent.first.ctx, parent.ctx)
        self.assertEqual((parent.second.ctx.n, parent.second.ctx.rate), (2, 4))
        events.clear()
        parent.reset()
        self.assertEqual(events, [("first", "reset", 1), ("second", "reset", 1), ("parent", "reset", 1, 1)])
        self.assertEqual(parent.count, 1)

    def test_registration_remains_closed_after_local_setup_and_in_reset(self):
        test = self
        rejected = []

        class Parent(Block):
            def build(self):
                self.worker = self.child("worker", Block())

            def setup(self, ctx):
                self.worker.setup(ctx)
                with test.assertRaises(RegistrationError):
                    self.worker.child("too late", Block())
                rejected.append("configured child")

            def reset(self):
                with test.assertRaises(RegistrationError):
                    self.child("during reset", Block())
                rejected.append("reset")

        parent = Parent()
        parent.setup(context())
        parent.reset()
        self.assertEqual(rejected, ["configured child", "reset", "reset"])
        self.assertEqual(list(parent.children), ["worker"])
        self.assertEqual(dict(parent.worker.children), {})

    def test_subtree_reset_preserves_other_state_configuration_and_storage(self):
        class Node(Block):
            count: int = State(default=1)
            history: Array = State(N, default=0)
            y: Array = Out(N)
            gain: float = Control(default=1.0)

            def process(self):
                self.count += 1
                self.history += self.gain
                self.y[:] = self.history

        root = Node()
        branch = root.child("branch", Node())
        leaf = branch.child("leaf", Node())
        sibling = root.child("sibling", Node())
        root.setup(context())
        nodes = (root, branch, leaf, sibling)
        arrays_before = [(node.history, node.y) for node in nodes]
        results = [node.run() for node in nodes]
        branch.gain = 3.0
        branch.reset()
        self.assertEqual([node.count for node in nodes], [2, 1, 1, 2])
        self.assertEqual(branch.gain, 3.0)
        for node, (state, output) in zip(nodes, arrays_before):
            self.assertIs(node.history, state)
            self.assertIs(node.y, output)
            np.testing.assert_array_equal(output, np.ones(4))
        np.testing.assert_array_equal(branch.history, np.zeros(4))
        np.testing.assert_array_equal(leaf.history, np.zeros(4))
        np.testing.assert_array_equal(root.history, np.ones(4))
        for node, result in zip(nodes, results):
            self.assertIs(node.run(), result)

    def test_prebuilt_subtree_stays_unready_until_root_finishes(self):
        test = self
        attempts = []

        class Worker(Block):
            count: int = State(default=0)

            def process(self):
                self.count += 1

        class Branch(Block):
            def build(self):
                self.worker = self.child("worker", Worker())

        class Root(Block):
            supplied: Block = Param()

            def build(self):
                self.branch = self.child("branch", self.supplied)

            def setup(self, ctx):
                self.branch.setup(ctx)
                self.check_pending()

            def reset(self):
                self.check_pending()

            def check_pending(self):
                for node in (self.branch, self.branch.worker):
                    for operation in (node.run, node.reset):
                        with test.assertRaises(SetupError):
                            operation()
                        attempts.append(operation)

        branch = Branch()  # Its descendant exists before the root owns it.
        root = Root(supplied=branch)
        root.setup(context())
        self.assertEqual(len(attempts), 8)
        self.assertEqual(branch.worker.count, 0)
        branch.worker.run()
        self.assertEqual(branch.worker.count, 1)
        branch.reset()
        self.assertEqual(branch.worker.count, 0)

    def test_registered_child_setup_is_parent_owned_and_once_only(self):
        parent = Block()
        child = parent.child("child", Block())
        with self.assertRaisesRegex(SetupError, "parent"):
            child.setup(context())
        with self.assertRaises(SetupError):
            child.reset()
        parent.setup(context())
        self.assertIs(child.ctx, parent.ctx)
        for node in (parent, child):
            with self.subTest(node=node), self.assertRaises(SetupError):
                node.setup(context())

    def test_ctx_and_child_dependent_axis_errors_identify_the_child(self):
        class Parent(Block):
            y: Array = Out(Axis("width"))

            def build(self):
                self.worker = self.child("sample/~worker", Block())

            @property
            def width(self):
                return self.worker.ctx.n

        parent = Parent()
        with self.assertRaisesRegex(SetupError, "Parent.*ctx"):
            _ = parent.ctx
        with self.assertRaisesRegex(SetupError, r"y.*axis width.*sample~1~0worker.*ctx") as caught:
            parent.setup(context())
        self.assertIsInstance(caught.exception.__cause__, SetupError)
        for node in (parent, parent.worker):
            with self.assertRaises(SetupError):
                node.run()

    def test_ctx_uses_public_setup_error_and_rejects_assignment(self):
        block = Block()
        with self.assertRaises(SetupError) as caught:
            _ = block.ctx
        self.assertIs(type(caught.exception), SetupError)
        with self.assertRaisesRegex(SetupError, "managed by setup"):
            block.ctx = context()
        configured = context()
        block.setup(configured)
        with self.assertRaisesRegex(SetupError, "managed by setup"):
            block.ctx = configured.replace(n=2)
        self.assertIs(block.ctx, configured)

    def test_parent_reset_hook_cannot_reset_a_child_during_setup(self):
        class Parent(Block):
            def build(self):
                self.worker = self.child("worker", Block())

            def reset(self):
                self.worker.reset()

        parent = Parent()
        with self.assertRaisesRegex(SetupError, r"parent reset hooks must not call child.reset"):
            parent.setup(context())
        with self.assertRaises(SetupError):
            parent.worker.run()

    def test_child_dependent_axes_accept_explicit_same_or_derived_context(self):
        class Worker(Block):
            width: int = Param(default=3)

        class Parent(Block):
            derived: bool = Param(default=False)
            y: Array = Out(Axis("width"))

            def build(self):
                self.worker = self.child("worker", Worker())

            def setup(self, ctx):
                self.worker.setup(ctx.replace(n=2) if self.derived else ctx)

            @property
            def width(self):
                return self.worker.ctx.n

        for derived, width in [(False, 4), (True, 2)]:
            parent = Parent(derived=derived)
            parent.setup(context())
            self.assertEqual(parent.y.shape, (width,))

        class FromParam(Parent):
            def setup(self, ctx):
                pass

            @property
            def width(self):
                return self.worker.width

        parent = FromParam()
        parent.setup(context())
        self.assertEqual(parent.y.shape, (3,))

    def test_unavailable_child_state_in_axis_preserves_the_cause(self):
        class Worker(Block):
            width: int = State(default=3)

        class Parent(Block):
            y: Array = Out(Axis("width"))

            def build(self):
                self.worker = self.child("worker", Worker())

            @property
            def width(self):
                return self.worker.width

        with self.assertRaisesRegex(SetupError, "axis width.*root/worker.*width") as caught:
            Parent().setup(context())
        self.assertIsInstance(caught.exception.__cause__, UnavailablePortError)

    def test_author_setup_error_in_axis_property_is_not_relabelled(self):
        error = SetupError("author validation")

        class Parent(Block):
            y: Array = Out(Axis("width"))

            @property
            def width(self):
                raise error

        with self.assertRaises(SetupError) as caught:
            Parent().setup(context())
        self.assertIs(caught.exception, error)

    def test_setup_failure_invalidates_all_nodes_even_if_child_error_is_caught(self):
        for catch_child_error in (False, True):
            error = ValueError("child setup failure")

            class Bad(Block):
                def setup(self, ctx):
                    raise error

            class Parent(Block):
                def build(self):
                    self.good = self.child("good", Block())
                    self.bad = self.child("bad", Bad())
                    self.pending = self.child("pending", Block())

                def setup(self, ctx):
                    self.good.setup(ctx)
                    if catch_child_error:
                        try:
                            self.bad.setup(ctx)
                        except ValueError:
                            pass

            parent = Parent()
            expected = SetupError if catch_child_error else ValueError
            with self.subTest(caught=catch_child_error), self.assertRaises(expected):
                parent.setup(context())
            for node in (parent, *parent.children.values()):
                for operation in (node.run, node.reset, lambda: node.setup(context())):
                    with self.assertRaises(SetupError):
                        operation()
                with self.assertRaisesRegex(RegistrationError, "after setup failure; construct a new tree"):
                    node.child("new", Block())
                with self.assertRaisesRegex(WiringError, "after setup failure; construct a new tree"):
                    node.connect(node, "x", node, "x")

    def test_initial_reset_failure_keeps_already_configured_children_unready(self):
        error = RuntimeError("initial reset failure")

        class Parent(Block):
            def build(self):
                self.worker = self.child("worker", Gain())

            def reset(self):
                raise error

        parent = Parent()
        with self.assertRaises(RuntimeError) as caught:
            parent.setup(context())
        self.assertIs(caught.exception, error)
        for node in (parent, parent.worker):
            for operation in (node.run, node.reset):
                with self.assertRaises(SetupError):
                    operation()

    def test_inherited_hooks_unwind_guards_and_do_not_duplicate_child_resets(self):
        events = []
        error = RuntimeError("reset failure")

        class Child(Block):
            def reset(self):
                events.append("child")

        class Base(Block):
            fail: bool = Control(default=False)

            def build(self):
                self.worker = self.child("worker", Child())

            def setup(self, ctx):
                events.append("base setup")

            def reset(self):
                events.append("base reset")
                if self.fail:
                    raise error

        class Derived(Base):
            def setup(self, ctx):
                super().setup(ctx)
                events.append("derived setup")

            def reset(self):
                super().reset()
                events.append("derived reset")

        parent = Derived()
        parent.setup(context())
        self.assertEqual(events, ["base setup", "derived setup", "child", "base reset", "derived reset"])
        events.clear()
        parent.fail = True
        with self.assertRaises(RuntimeError) as caught:
            parent.reset()
        self.assertIs(caught.exception, error)
        parent.fail = False
        parent.reset()
        self.assertEqual(events, ["child", "base reset", "child", "base reset", "derived reset"])

        failed = Derived(fail=True)
        with self.assertRaises(RuntimeError):
            failed.setup(context())
        with self.assertRaises(SetupError):
            failed.reset()
        with self.assertRaises(SetupError):
            failed.setup(context())

    def test_readiness_is_per_consumer_and_execution_remains_caller_ordered(self):
        class Parent(Block):
            def build(self):
                self.used = self.child("used", Gain(gain=2.0))
                self.unused = self.child("unused", Gain())

            def process(self):
                self.used.run()

        parent = Parent()
        parent.used.x = np.arange(4.0).reshape(1, 4)
        parent.setup(context())
        result = parent.used.run()
        np.testing.assert_array_equal(result, [[0, 2, 4, 6]])
        self.assertIsNone(parent.run())
        with self.assertRaises(UnavailablePortError):
            parent.unused.run()
        value = parent.used.x
        parent.reset()
        self.assertIs(parent.used.x, value)
        self.assertIs(parent.used.run(), result)
