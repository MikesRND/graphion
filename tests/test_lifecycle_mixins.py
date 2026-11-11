"""Lifecycle hooks inherited from non-Block mixins use framework entry points."""

import unittest
from functools import wraps

import numpy as np

from graphion import Array, Block, DeclarationError, N, Out, Param, SetupError, State
from tests.test_connections import context


class LifecycleMixinTests(unittest.TestCase):
    def test_mixin_hooks_without_super_and_inherited_wrappers(self):
        events = []

        class Hooks:
            def build(self):
                events.append("build")
                self.worker = self.child("worker", Block())

            def setup(self, ctx):
                events.append("setup")
                self.seen_ctx = self.ctx
                self.bind_output("y", np.zeros(ctx.n))

            def reset(self):
                events.append("reset")
                self.count += 1
                self.memory += 2

        class Model(Hooks, Block):
            y: Array = Out(N)
            count: int = State(default=3)
            memory: Array = State(N, default=1.0)

        class Inherited(Model):
            pass

        class Descendant(Inherited):
            pass

        for name in ("build", "setup", "reset"):
            self.assertIs(getattr(Descendant, name), getattr(Model, name))
        for cls in (Model, Descendant):
            with self.subTest(cls=cls):
                events.clear()
                model = cls()
                self.assertIs(model.children["worker"], model.worker)
                self.assertEqual(events, ["build"])
                self.assertEqual(model.describe()["topology"], "open")
                with self.assertRaises(SetupError):
                    model.reset()
                model.build()
                model.build()
                self.assertEqual(events, ["build"])
                ctx = context()
                model.setup(ctx)
                self.assertIs(model.seen_ctx, ctx)
                self.assertIs(model.worker.ctx, ctx)
                self.assertEqual(events, ["build", "setup", "reset"])
                self.assertEqual(model.count, 4)
                np.testing.assert_array_equal(model.memory, np.full(4, 3.0))
                output, memory = model.y, model.memory
                model.count = 90
                model.memory.fill(90)
                model.reset()
                self.assertEqual(model.count, 4)
                np.testing.assert_array_equal(model.memory, np.full(4, 3.0))
                self.assertIs(model.y, output)
                self.assertIs(model.memory, memory)
                self.assertEqual(events, ["build", "setup", "reset", "reset"])
                model.build()
                with self.assertRaises(SetupError):
                    model.setup(ctx)
                self.assertEqual(events, ["build", "setup", "reset", "reset"])

    def test_cooperative_mixin_hooks_reach_framework_base(self):
        events = []

        class Hooks:
            def build(self):
                super().build()
                events.append("build")
                self.worker = self.child("worker", Block())

            def setup(self, ctx):
                super().setup(ctx)
                events.append("setup")
                self.seen_ctx = self.ctx
                self.bind_output("y", np.zeros(ctx.n))

            def reset(self):
                super().reset()
                events.append("reset")
                self.count += 1

        class Model(Hooks, Block):
            y: Array = Out(N)
            count: int = State(default=3)

        model = Model()
        self.assertEqual(model.describe()["topology"], "open")
        model.build()
        model.setup(context())
        self.assertEqual(events, ["build", "setup", "reset"])
        self.assertIs(model.seen_ctx, model.ctx)
        self.assertEqual(model.count, 4)
        model.count = 90
        model.reset()
        self.assertEqual(model.count, 4)
        self.assertEqual(events, ["build", "setup", "reset", "reset"])

    def test_cooperative_mixin_hooks_reach_wrapped_author_bodies_once(self):
        events = []

        class Worker(Block):
            def reset(self):
                events.append("worker reset")

        class Base(Block):
            y: Array = Out(N)
            count: int = State(default=3)

            def build(self):
                events.append("base build")
                self.worker = self.child("worker", Worker())

            def setup(self, ctx):
                events.append("base setup")

            def reset(self):
                events.append("base reset")
                self.count += 1

        class Hooks:
            def build(self):
                super().build()
                events.append("mixin build")

            def setup(self, ctx):
                super().setup(ctx)
                events.append("mixin setup")
                self.bind_output("y", np.zeros(ctx.n))

            def reset(self):
                super().reset()
                events.append("mixin reset")
                self.count += 1

        class Model(Hooks, Base):
            pass

        model = Model()
        model.build()
        model.setup(context())
        self.assertEqual(
            events,
            ["base build", "mixin build", "base setup", "mixin setup", "worker reset", "base reset", "mixin reset"],
        )
        self.assertEqual(model.count, 5)
        events.clear()
        model.count = 90
        model.reset()
        self.assertEqual(events, ["worker reset", "base reset", "mixin reset"])
        self.assertEqual(model.count, 5)

    def test_decorated_override_is_wrapped_despite_copied_wrapper_metadata(self):
        events = []

        class Base(Block):
            count: int = State(default=0)

            def build(self):
                events.append("base build")

            def setup(self, ctx):
                events.append("base setup")

            def reset(self):
                events.append("base reset")
                self.count += 1

        class Derived(Base):
            @wraps(Base.build)
            def build(self):
                super().build()
                events.append("derived build")
                self.worker = self.child("worker", Block())

            @wraps(Base.setup)
            def setup(self, ctx):
                self.seen_ctx = self.ctx
                super().setup(ctx)
                events.append("derived setup")

            @wraps(Base.reset)
            def reset(self):
                super().reset()
                events.append("derived reset")
                self.count += 1

        model = Derived()
        model.build()
        model.setup(context())
        self.assertEqual(
            events,
            ["base build", "derived build", "base setup", "derived setup", "base reset", "derived reset"],
        )
        self.assertEqual(model.count, 2)
        events.clear()
        model.count = 90
        model.reset()
        self.assertEqual(model.count, 2)
        self.assertEqual(events, ["base reset", "derived reset"])

    def test_framework_owned_names_are_rejected_in_mixins_in_either_mro_order(self):
        names = (
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
            "port_spec",
        )
        for name in names:
            mixin = type("ProtectedMixin", (), {name: lambda self: None})
            inherited = type("InheritedMixin", (mixin,), {})
            for candidate in (mixin, inherited):
                for bases in ((candidate, Block), (Block, candidate)):
                    with self.subTest(name=name, bases=bases):
                        with self.assertRaisesRegex(DeclarationError, f"ProtectedMixin.*framework-owned.*{name}"):
                            type("Invalid", bases, {})

    def test_protected_mixin_descriptors_are_rejected_without_evaluation(self):
        class Descriptor:
            def __get__(self, instance, owner=None):
                raise AssertionError("protected descriptor was evaluated")

        for name in ("schema", "ctx", "children"):
            mixin = type("ProtectedMixin", (), {name: Descriptor()})
            for bases in ((mixin, Block), (Block, mixin)):
                with self.subTest(name=name, bases=bases):
                    with self.assertRaisesRegex(DeclarationError, f"framework-owned.*{name}"):
                        type("Invalid", bases, {})

    def test_generated_schemas_remain_valid_through_multiple_block_generations(self):
        class Helper:
            def label(self):
                return "helper"

        class Parent(Helper, Block):
            amount: int = Param(default=1)
            y: Array = Out(N)

        class Child(Parent):
            amount: int = Param(default=2)

        class Grandchild(Child):
            count: int = State(default=3)

        self.assertEqual(list(Parent.schema), ["amount", "y"])
        self.assertEqual(list(Child.schema), ["amount", "y"])
        self.assertEqual(list(Grandchild.schema), ["amount", "y", "count"])
        self.assertIsNot(Child.schema, Parent.schema)
        self.assertIsNot(Grandchild.schema, Child.schema)
        model = Grandchild()
        self.assertEqual(model.label(), "helper")
        model.build()
        model.setup(context())
        self.assertEqual(model.amount, 2)
        self.assertEqual(model.count, 3)
        self.assertEqual(model.y.shape, (4,))

    def test_none_setup_and_reset_disable_author_bodies_without_disabling_framework(self):
        events = []

        class Base(Block):
            y: Array = Out(N)
            count: int = State(default=3)

            def setup(self, ctx):
                events.append("setup")

            def reset(self):
                events.append("reset")
                self.count += 1

        for name in ("setup", "reset"):
            mixin = type("DisabledHookMixin", (), {name: None})
            disabled_classes = (
                type("Disabled", (Base,), {name: None}),
                type("DisabledViaMixin", (mixin, Base), {}),
            )
            for disabled in disabled_classes:
                inherited = type("Inherited", (disabled,), {})
                for cls in (disabled, inherited):
                    with self.subTest(hook=name, cls=cls):
                        events.clear()
                        model = cls()
                        with self.assertRaises(SetupError):
                            model.reset()
                        model.setup(context())
                        self.assertEqual(events, ["reset"] if name == "setup" else ["setup"])
                        self.assertEqual(model.count, 4 if name == "setup" else 3)
                        self.assertEqual(model.y.shape, (4,))
                        self.assertEqual(model.describe()["topology"], "complete")
                        self.assertIsNotNone(model.describe()["blocks"][0]["storage"])
                        events.clear()
                        model.count = 90
                        output = model.y
                        model.reset()
                        self.assertEqual(events, ["reset"] if name == "setup" else [])
                        self.assertEqual(model.count, 4 if name == "setup" else 3)
                        self.assertIs(model.y, output)
                        with self.assertRaises(SetupError):
                            model.setup(context())
