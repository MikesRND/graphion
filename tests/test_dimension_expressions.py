"""Exact declaration arithmetic, ownership, and resolved port inspection."""

import json
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction
from unittest.mock import patch

import numpy as np

from graphion import Array, Block, C, Ctx, DeclarationError, In, N, Out, Param, PortSpec, SetupError, State
from graphion.profile.numpy_frame import _dimensions
from tests.test_declarations import module_with


class RequiredSize(Block):
    factor: int = Param()
    x: Array = In(C, N)
    y: Array = Out(C, N / factor)

    def process(self) -> None:
        self.y[...] = self.x[:, :: self.factor]


class OverriddenSize(RequiredSize):
    factor: int = Param(default=2)
    y: Array = Out(C, N / RequiredSize.factor)


def context(n: int = 6) -> Ctx:
    return Ctx(rate=48_000, n=n, channels=2)


class DimensionExpressionTests(unittest.TestCase):
    def test_copied_tokens_and_expressions_preserve_symbol_meaning(self) -> None:
        for token, expected in ((C, 3), (N, 8)):
            copied = deepcopy(token)
            self.assertIsNot(copied, token)
            self.assertEqual(copied, token)
            for expression in (copied * 2, deepcopy(token * 2), _dimensions.SizeToken(token.name) * 2):
                with self.subTest(token=token, expression=expression):

                    class Model(Block):
                        y: Array = Out(expression)

                    block = Model()
                    block.setup(Ctx(rate=1, channels=3, n=8))
                    self.assertEqual(block.port_spec("y").shape, (expected * 2,))
                    dimension = block.describe()["blocks"][0]["declarations"][0]
                    self.assertEqual(dimension["declared_shape"][0]["left"], {"kind": "symbol", "name": token.name})
                    self.assertEqual(dimension["shape"], [{"kind": "resolved", "value": expected * 2}])

    def test_unknown_size_token_does_not_default_to_frame_length(self) -> None:
        class Model(Block):
            y: Array = Out(_dimensions.SizeToken("other") * 2)

        with self.assertRaisesRegex(SetupError, "root.*y:.*unknown size token 'other'"):
            Model().setup(context())

    def test_divisibility_error_identifies_expression_and_result(self) -> None:
        block = RequiredSize(factor=3)
        with self.assertRaisesRegex(SetupError, r"root.*y:.*\(N / factor\).*4/3.*non-negative integer"):
            block.setup(context(4))

    def test_exact_intermediates_and_reflected_multiplication(self) -> None:
        class Model(Block):
            m: int = Param(default=2)
            n: int = Param(default=2)
            ratio: Fraction = Param(default=Fraction(2, 3))
            a: Array = Out(C * 2, N / n * m)
            b: Array = Out(C, N * m / n)
            c: Array = Out(C, ratio * N)
            huge: Array = Out(N * 10**400 / 10**400)
            empty: Array = Out(N * 0)
            held: Array = State(C, N / n * m, default=0)

        model = Model()
        model.setup(context(3))
        self.assertEqual(model.port_spec("a").shape, (4, 3))
        self.assertEqual(model.port_spec("b").shape, (2, 3))
        self.assertEqual(model.port_spec("c").shape, (2, 2))
        self.assertEqual(model.port_spec("huge").shape, (3,))
        self.assertEqual(model.port_spec("empty").shape, (0,))
        np.testing.assert_array_equal(model.held, np.zeros((2, 3)))

    def test_literal_division_and_fraction_have_identical_results(self) -> None:
        class Model(Block):
            a: Array = Out(N * 3 / 2)
            b: Array = Out(N * Fraction(3, 2))
            c: Array = Out(N * np.int64(3) / np.int64(2))

        model = Model()
        model.setup(context(4))
        self.assertEqual([model.port_spec(name).shape for name in ("a", "b", "c")], [(6,)] * 3)

    def test_invalid_literals_fail_during_class_definition(self) -> None:
        for operand in ("(3 / 2)", "1.5", "True", "np.bool_(False)", "np.float64(2)"):
            with self.subTest(operand=operand), self.assertRaises(DeclarationError):
                with module_with(f"""
import numpy as np
from graphion import Array, Block, N, Out
class Model(Block):
    y: Array = Out(N * {operand})
"""):
                    self.fail("invalid expression accepted")

    def test_invalid_configured_values_fail_at_setup(self) -> None:
        for value, message in (
            (0, "divides by zero"),
            (4, "non-negative integer"),
            (-2, "non-negative integer"),
            (2.0, "integers or Fractions"),
            (True, "bool"),
            ("2", "integers or Fractions"),
        ):
            with self.subTest(value=value), self.assertRaisesRegex(SetupError, message):
                # Runtime configuration validation is exercised independently of typing.
                model = RequiredSize(**{"factor": value})  # type: ignore[arg-type]
                model.setup(context())

    def test_fraction_param_and_instance_overrides_are_independent(self) -> None:
        class Model(Block):
            ratio: Fraction = Param(default=Fraction(1, 2))
            y: Array = Out(C, N * ratio)

        small, large = Model(), Model(ratio=Fraction(3, 2))
        small.setup(context())
        large.setup(context())
        self.assertEqual(small.port_spec("y").shape, (2, 3))
        self.assertEqual(large.port_spec("y").shape, (2, 9))

    def test_inherited_and_explicit_base_references_use_overrides(self) -> None:
        base_field = RequiredSize.schema["y"]
        base_axes = base_field.axes

        class Inherited(RequiredSize):
            factor: int = Param(default=3)

        base, explicit, inherited = RequiredSize(factor=6), OverriddenSize(), Inherited()
        for block in (base, explicit, inherited):
            block.setup(context())
        self.assertEqual([b.port_spec("y").shape for b in (base, explicit, inherited)], [(2, 1), (2, 3), (2, 2)])
        self.assertIs(RequiredSize.schema["y"], base_field)
        self.assertIs(base_field.axes, base_axes)
        self.assertIs(Inherited.schema["y"], base_field)

    def test_unrelated_and_unbound_references_fail(self) -> None:
        for body in (
            "factor: int = Param(default=2)\n    y: Array = Out(N / Other.factor)",
            "y: Array = Out(N / outside)",
            "y: Array = Out(Other.schema['y'].axes[0] * 2)",
        ):
            with self.subTest(body=body), self.assertRaisesRegex(DeclarationError, "belong"):
                with module_with(f"""
from graphion import Array, Block, N, Out, Param
class Other(Block):
    factor: int = Param(default=2)
    y: Array = Out(N / factor)
outside = Param(default=2)
class Model(Block):
    {body}
"""):
                    self.fail("foreign reference accepted")

    def test_aliases_bare_params_and_non_params_fail(self) -> None:
        bodies = (
            ("a: int = Param(default=2)\n    b: int = a", "two names"),
            ("a: int = Param(default=2)\n    y: Array = Out(a)", "bare declaration"),
            ("a: int = Control(default=2)\n    y: Array = Out(N * a)", "only Params"),
            ("a: int = State(default=2)\n    y: Array = Out(N * a)", "only Params"),
            ("a: Array = In(N)\n    y: Array = Out(N * a)", "only Params"),
        )
        for body, message in bodies:
            with self.subTest(body=body), self.assertRaisesRegex(DeclarationError, message):
                with module_with(f"""
from graphion import Array, Block, Control, In, N, Out, Param, State
class Model(Block):
    {body}
"""):
                    self.fail("invalid dimension accepted")

    def test_only_size_tokens_and_expressions_have_arithmetic(self) -> None:
        for expression in ('Axis("width") * 2', "Sample * 2", "Param(default=2) + 1"):
            with self.subTest(expression=expression), self.assertRaises(TypeError):
                with module_with(f"""
from graphion import Axis, Param, Sample
value = {expression}
"""):
                    self.fail("unsupported arithmetic accepted")

    def test_description_preserves_expression_and_exact_fraction(self) -> None:
        class Model(Block):
            ratio: Fraction = Param(default=Fraction(2, 3))
            y: Array = Out(C, N * ratio / Fraction(1, 2))

        model = Model()
        with patch.object(_dimensions, "resolve", side_effect=AssertionError("early evaluation")):
            before = model.describe()
        declared = before["blocks"][0]["declarations"][1]["declared_shape"]
        expected = {
            "kind": "expression",
            "operator": "divide",
            "left": {
                "kind": "expression",
                "operator": "multiply",
                "left": {"kind": "symbol", "name": "N"},
                "right": {"kind": "param", "name": "ratio"},
            },
            "right": {"kind": "fraction", "numerator": 1, "denominator": 2},
        }
        self.assertEqual(declared[1], expected)
        self.assertEqual(
            before["blocks"][0]["parameters"]["ratio"], {"kind": "fraction", "numerator": 2, "denominator": 3}
        )
        model.setup(context())
        with patch.object(_dimensions, "resolve", side_effect=AssertionError("reevaluated")):
            after = model.describe()
        declaration = after["blocks"][0]["declarations"][1]
        self.assertEqual(declaration["declared_shape"], declared)
        self.assertEqual(declaration["shape"], [{"kind": "resolved", "value": 2}, {"kind": "resolved", "value": 8}])
        self.assertEqual(after["unresolved"], [])
        self.assertEqual(before["blocks"][0]["declarations"][1]["shape"], declared)
        self.assertEqual(json.loads(json.dumps(after, allow_nan=False)), after)

    def test_unresolved_shape_and_declaration_are_independent_trees(self) -> None:
        block = RequiredSize(factor=2)
        snapshot = block.describe()
        port = snapshot["blocks"][0]["declarations"][2]
        original = deepcopy(port["declared_shape"])
        port["shape"][1]["right"]["name"] = "changed"
        port["shape"].append({"kind": "resolved", "value": 99})
        self.assertEqual(port["declared_shape"], original)
        port["declared_shape"][1]["left"]["name"] = "different"
        self.assertEqual(port["shape"][1]["left"]["name"], "N")
        self.assertEqual(block.describe()["blocks"][0]["declarations"][2]["shape"], original)

    def test_processing_reset_and_inspection_do_not_evaluate_expressions(self) -> None:
        block = OverriddenSize()
        block.setup(context())
        block.x = np.arange(12.0).reshape(2, 6)
        with patch.object(_dimensions, "resolve", side_effect=AssertionError("per-run resolution")):
            np.testing.assert_array_equal(block.run(), block.x[:, ::2])
            block.reset()
            np.testing.assert_array_equal(block.run(), block.x[:, ::2])
            block.describe()
            self.assertEqual(block.port_spec("y").shape, (2, 3))


class PortSpecTests(unittest.TestCase):
    def test_scalar_port_spec_and_explicit_dtype(self) -> None:
        class Model(Block):
            x: Array = In(dtype=np.int16)
            memory: Array = State(N, default=0)

        block = Model()
        block.setup(context())
        self.assertEqual(block.port_spec("x"), PortSpec((), np.dtype(np.int16)))
        with self.assertRaisesRegex(SetupError, "not a port"):
            block.port_spec("memory")

    def test_specs_are_immutable_cached_and_available_without_inputs(self) -> None:
        block = OverriddenSize()
        with self.assertRaisesRegex(SetupError, "local setup"):
            block.port_spec("x")
        block.setup(context())
        spec = block.port_spec("x")
        self.assertIsInstance(spec, PortSpec)
        self.assertEqual(spec.shape, (2, 6))
        self.assertEqual(spec.dtype, np.dtype(np.float64))
        self.assertIs(block.port_spec("x"), spec)
        with self.assertRaises(FrozenInstanceError):
            setattr(spec, "shape", (1,))
        for name in ("missing", "factor"):
            with self.subTest(name=name), self.assertRaisesRegex(SetupError, "not a port"):
                block.port_spec(name)

    def test_child_specs_before_tree_finalization_and_forwarded_outputs(self) -> None:
        class Parent(Block):
            x: Array = In(C, N)
            y: Array = Out(C, N / 2, dtype=np.float64)

            def build(self) -> None:
                self.child_block = self.child("child", OverriddenSize())
                self.connect(self, "x", self.child_block, "x")
                self.connect(self.child_block, "y", self, "y")

            def setup(self, ctx: Ctx) -> None:
                self.child_block.setup(ctx)
                self.child_spec = self.child_block.port_spec("y")

        parent = Parent()
        parent.setup(context())
        self.assertEqual(parent.child_spec.shape, (2, 3))
        self.assertEqual(parent.port_spec("y"), parent.child_spec)
        self.assertEqual(parent.port_spec("x").shape, (2, 6))

    def test_failed_local_setup_does_not_publish_specs(self) -> None:
        class Model(Block):
            x: Array = In(C, N)

            def reset(self) -> None:
                raise ValueError("reset failed")

        block = Model()
        with self.assertRaisesRegex(ValueError, "reset failed"):
            block.setup(context())
        with self.assertRaisesRegex(SetupError, "local setup"):
            block.port_spec("x")

    def test_initial_reset_cannot_read_own_specs_but_later_reset_can(self) -> None:
        class Model(Block):
            y: Array = Out(C, N)

            def reset(self) -> None:
                try:
                    self.port_spec("y")
                except SetupError:
                    self.spec_unavailable = True
                else:
                    self.spec_unavailable = False

        block = Model()
        block.setup(context())
        self.assertTrue(block.spec_unavailable)
        block.reset()
        self.assertFalse(block.spec_unavailable)

    def test_child_specs_survive_ancestor_failure(self) -> None:
        class Parent(Block):
            def build(self) -> None:
                self.child_block = self.child("child", OverriddenSize())

            def setup(self, ctx: Ctx) -> None:
                self.child_block.setup(ctx)
                raise ValueError("parent failed")

        parent = Parent()
        with self.assertRaisesRegex(ValueError, "parent failed"):
            parent.setup(context())
        self.assertEqual(parent.child_block.port_spec("y").shape, (2, 3))
        with self.assertRaises(SetupError):
            parent.child_block.run()

    def test_port_spec_is_a_reserved_member(self) -> None:
        for body in ("port_spec: int = Param(default=1)", "def port_spec(self): pass"):
            with self.subTest(body=body), self.assertRaises(DeclarationError):
                with module_with(f"""
from graphion import Block, Param
class Model(Block):
    {body}
"""):
                    self.fail("reserved member accepted")
