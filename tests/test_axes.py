import unittest

import numpy as np

from graphion import Array, Axis, Block, Ctx, Out, Param, SetupError


class AxisTests(unittest.TestCase):
    def test_named_axes_use_parameters_properties_and_setup_attributes(self):
        class Model(Block):
            width: int = Param(default=2)
            a: Array = Out(Axis("width"))
            b: Array = Out(Axis("computed"))
            c: Array = Out(Axis("stored"))

            @property
            def computed(self):
                return self.ctx.n

            def setup(self, ctx):
                self.stored = ctx.n + 1

        model = Model()
        model.setup(Ctx(rate=1, n=4, channels=1))
        self.assertEqual((model.a.shape, model.b.shape, model.c.shape), ((2,), (4,), (5,)))

    def test_missing_axis_and_dynamic_lookup_are_rejected(self):
        calls = []

        class Missing(Block):
            y: Array = Out(Axis("width"))

        class Dynamic(Missing):
            def __getattr__(self, name):
                calls.append(name)
                return 4

        class ClassConstant(Missing):
            width = 4

        for model in (Missing(), Dynamic(), ClassConstant()):
            with self.subTest(model=type(model)), self.assertRaisesRegex(SetupError, "y.*axis width"):
                model.setup(Ctx(rate=1, n=4, channels=1))
        self.assertEqual(calls, [])

    def test_existing_axis_property_exceptions_propagate_unchanged(self):
        for error in (AttributeError("failure inside property"), ValueError("author failure")):

            class Model(Block):
                y: Array = Out(Axis("width"))

                @property
                def width(self):
                    raise error

                def __getattr__(self, name):
                    # A property AttributeError must not fall through here.
                    return 4

            with self.subTest(error=error):
                with self.assertRaises(type(error)) as caught:
                    Model().setup(Ctx(rate=1, n=4, channels=1))
                self.assertIs(caught.exception, error)

    def test_slotted_setup_attribute_is_supported_and_unset_slot_is_missing(self):
        class Slotted(Block):
            __slots__ = ("width",)
            y: Array = Out(Axis("width"))

        class Ready(Slotted):
            def setup(self, ctx):
                self.width = ctx.n

            def process(self):
                self.y.fill(2)

        model = Ready()
        model.setup(Ctx(rate=1, n=4, channels=1))
        np.testing.assert_array_equal(model.run(), [2, 2, 2, 2])
        with self.assertRaisesRegex(SetupError, "y.*axis width"):
            Slotted().setup(Ctx(rate=1, n=4, channels=1))
