import sys
import types
import unittest
from contextlib import contextmanager
from dataclasses import FrozenInstanceError

import numpy as np
from numpy.typing import NDArray

from graphion import (
    MISSING,
    Array,
    Axis,
    Block,
    C,
    Control,
    Ctx,
    DeclarationError,
    In,
    N,
    Out,
    Param,
    ParamError,
    Sample,
    State,
    StorageError,
)
from graphion.dsp.blocks import Gain


@contextmanager
def module_with(source, name="_graphion_annotation_fixture"):
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        exec(compile(source, f"<{name}>", "exec"), module.__dict__)
        yield module
    finally:
        del sys.modules[name]


class DeclarationTests(unittest.TestCase):
    def test_schema_equality_is_identity_based(self):
        for default in (np.array([1.0]), np.array([1.0, 2.0]), np.empty(0)):
            records = []
            for _ in range(2):
                cls = type(
                    "Model",
                    (Block,),
                    {
                        "__annotations__": {"value": Array},
                        "value": Param(default=default.copy()),
                    },
                )
                records.append(cls.schema["value"])
            left, right = records
            self.assertIs(left == left, True)
            self.assertIs(left == right, False)
            self.assertIs(left != right, True)
            self.assertEqual(len({left, right}), 2)
            # Content comparison is explicit, not record equality.
            self.assertEqual((left.kind, left.value_type), (right.kind, right.value_type))
            np.testing.assert_array_equal(left.default, right.default)

    def test_missing_sentinel_has_stable_representation(self):
        self.assertEqual(repr(MISSING), "MISSING")
        self.assertEqual(str(MISSING), "MISSING")
        self.assertIsNot(MISSING, None)

    def test_schema_records_and_default_identity(self):
        default = np.array([1.0, 2.0])

        class Model(Block):
            required: int = Param()
            optional: object = Control(default=None)
            coefficients: Array = Param(default=default)
            count: int = State(default=0)
            memory: Array = State(C, N, default=0.0, dtype=np.float32)
            x: Array = In()
            y: Array = Out(C, Axis("width"))

        schema = Model.schema
        self.assertEqual(list(schema), ["required", "optional", "coefficients", "count", "memory", "x", "y"])
        for name, kind, hint, dtype, axes, value in (
            ("required", "Param", int, None, None, MISSING),
            ("optional", "Control", object, None, None, None),
            ("coefficients", "Param", Array, None, None, default),
            ("count", "State", int, None, None, 0),
            ("memory", "State", Array, np.float32, (C, N), 0.0),
            ("x", "In", Array, Sample, (), MISSING),
            ("y", "Out", Array, Sample, (C, Axis("width")), MISSING),
        ):
            field = schema[name]
            self.assertEqual(
                (field.name, field.kind, field.value_type, field.dtype, field.axes), (name, kind, hint, dtype, axes)
            )
            self.assertIs(field.default, value)
            with self.assertRaises(FrozenInstanceError):
                field.default = None
        with self.assertRaises(TypeError):
            schema["other"] = schema["x"]
        self.assertTrue(default.flags.writeable)
        a, b = Model(required=1), Model(required=2)
        default.fill(9)
        np.testing.assert_array_equal(a.coefficients, [1, 2])
        self.assertFalse(np.shares_memory(a.coefficients, b.coefficients))
        self.assertIs(schema["coefficients"].default, default)

    def test_field_override_is_complete_and_keeps_order(self):
        class Required(Gain):
            gain: float = Control()
            extra: int = Param(default=3)

        class Untouched(Required):
            pass

        self.assertEqual(list(Required.schema), ["gain", "x", "y", "extra"])
        self.assertIs(Required.schema["gain"].default, MISSING)
        self.assertIs(Untouched.schema["gain"], Required.schema["gain"])
        self.assertEqual(Gain().gain, 1.0)
        self.assertEqual(Untouched(gain=2).extra, 3)
        with self.assertRaises(ParamError):
            Required()

    def test_inherited_fields_cannot_be_silently_shadowed(self):
        for attributes in (
            {"gain": 2.0},
            {"gain": lambda self: 2.0},
            {"gain": property(lambda self: 2.0)},
            {"__annotations__": {"gain": float}},
            {"gain": Control(default=2.0)},
            {"gain": Param(default=2.0), "__annotations__": {"gain": float}},
        ):
            with self.subTest(attributes=attributes):
                with self.assertRaisesRegex(DeclarationError, "Shadow.gain"):
                    type("Shadow", (Gain,), attributes)

    def test_constructor_accepts_only_keyword_configuration(self):
        class Model(Block):
            factor: int = Param()
            gain: float = Control(default=1.0)
            count: int = State(default=0)
            x: Array = In(C, N)
            y: Array = Out(C, N)

        for kwargs in (
            {},
            {"factor": 2, "x": np.ones((1, 1))},
            {"factor": 2, "y": np.ones((1, 1))},
            {"factor": 2, "count": 1},
            {"factor": 2, "typo": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ParamError):
                Model(**kwargs)
        with self.assertRaises(TypeError):
            Model(2)
        model = Model(factor=2, gain="stored without coercion")
        self.assertEqual(model.gain, "stored without coercion")
        with self.assertRaises(ParamError):
            model.factor = 3

    def test_array_configuration_annotations_control_copying(self):
        for hint in (np.ndarray, Array, NDArray[np.float64]):
            with self.subTest(hint=hint):
                default = np.array([1.0, 2.0])
                model_type = type(
                    "Configuration",
                    (Block,),
                    {
                        "__annotations__": {"p": hint, "c": hint},
                        "p": Param(default=default),
                        "c": Control(default=default),
                    },
                )
                a, b = model_type(), model_type()
                self.assertFalse(a.p.flags.writeable)
                self.assertFalse(a.c.flags.writeable)
                for left, right in ((a.p, default), (a.c, default), (a.p, b.p), (a.c, b.c)):
                    self.assertFalse(np.shares_memory(left, right))
                replacement = np.array([3.0, 4.0])
                a.c = replacement
                replacement.fill(9)
                np.testing.assert_array_equal(a.c, [3.0, 4.0])
                with self.assertRaises(ValueError):
                    a.p[0] = 9
        for hint, default in ((Array | None, np.ones(2)), (list[Array], [np.ones(2)])):
            model_type = type(
                "AuthorOwned",
                (Block,),
                {
                    "__annotations__": {"p": hint},
                    "p": Param(default=default),
                },
            )
            self.assertIs(model_type().p, default)

    def test_ports_and_array_state_require_array_alias(self):
        for hint, field in (
            (float, In()),
            (np.ndarray, Out()),
            (NDArray[np.float64], In(N)),
            (float, State(N, default=0)),
            (Array, State(default=0)),
            (str, State(default="")),
            (object, State(default=object())),
        ):
            with self.subTest(hint=hint, kind=field.kind):
                with self.assertRaisesRegex(DeclarationError, "Mismatch.value"):
                    type("Mismatch", (Block,), {"__annotations__": {"value": hint}, "value": field})

    def test_invalid_array_configuration_defaults_fail_at_definition(self):
        invalid = (
            None,
            [[1], [2, 3]],
            {"key": 1},
            object(),
            np.array([1, None], dtype=object),
            np.array([(object(),)], dtype=[("item", object)]),
        )
        for factory in (Param, Control):
            for hint in (np.ndarray, Array, NDArray[np.float64]):
                for value in invalid:
                    with self.subTest(factory=factory, hint=hint, value=value):
                        with self.assertRaisesRegex(DeclarationError, "Invalid.value default"):
                            type(
                                "Invalid",
                                (Block,),
                                {
                                    "__annotations__": {"value": hint},
                                    "value": factory(default=value),
                                },
                            )

    def test_invalid_array_configuration_preserves_current_control(self):
        class Model(Block):
            p: Array = Param(default=[1.0, 2.0])
            c: Array = Control(default=[3.0, 4.0])

        model = Model()
        original = model.c
        for value in (None, [[1], [2, 3]], object(), np.array([None]), np.empty(2, dtype=[("item", object)])):
            with self.subTest(value=value):
                for name in ("p", "c"):
                    with self.assertRaisesRegex(ParamError, name):
                        Model(**{name: value})
                with self.assertRaisesRegex(ParamError, "c"):
                    model.c = value
                self.assertIs(model.c, original)
                np.testing.assert_array_equal(model.c, [3.0, 4.0])
        for value in ([], [[1, 2], [3, 4]], 1.5, np.array([1 + 2j])):
            with self.subTest(valid=value):
                model.c = value
                np.testing.assert_array_equal(model.c, np.asarray(value))
                self.assertFalse(model.c.dtype.hasobject)
        # Mutating a declaration's default does not bypass construction checks.
        Model.schema["p"].default.append([3, 4])
        with self.assertRaisesRegex(ParamError, "p"):
            Model()

    def test_numeric_scalar_state_types_and_no_coercion(self):
        for hint in (bool, int, float, complex, np.bool_, np.int64, np.float32, np.complex128):
            with self.subTest(hint=hint):
                model_type = type(
                    "Scalar",
                    (Block,),
                    {
                        "__annotations__": {"value": hint},
                        "value": State(default=1),
                    },
                )
                model = model_type()
                model.setup(Ctx(rate=1, n=1, channels=1))
                self.assertIs(type(model.value), int)
                model.value = 7
                model.reset()
                self.assertEqual(model.value, 1)

    def test_explicit_dtype_and_array_state_storage(self):
        class Model(Block):
            memory: Array = State(C, N, dtype=np.float32, default=2.0)
            y: Array = Out(C, N, dtype=np.float32)

            def process(self):
                self.memory *= 2
                np.copyto(self.y, self.memory)

        model = Model()
        schema = tuple(Model.schema.values())
        model.setup(Ctx(rate=1, n=2, channels=1, dtype=np.complex128))
        storage = model.memory
        self.assertEqual(model.y.dtype, np.float32)
        np.testing.assert_array_equal(model.run(), [[4.0, 4.0]])
        self.assertIs(model.memory, storage)
        with self.assertRaises(StorageError):
            model.memory = storage.copy()
        model.reset()
        np.testing.assert_array_equal(storage, 2)
        self.assertEqual(tuple(Model.schema.values()), schema)

    def test_invalid_field_arguments_raise_declaration_error(self):
        invalid = (
            lambda: Param(1),
            lambda: Param(dtype=float),
            lambda: Control(1),
            lambda: Control(dtype=float),
            lambda: In(default=1),
            lambda: Out(default=1),
            lambda: State(),
            lambda: State(N),
            lambda: State(default=0, dtype=float),
            lambda: In(unknown=True),
            lambda: Param(init=False),
            lambda: Control(init=False),
        )
        for factory in invalid:
            with self.subTest(factory=factory), self.assertRaises(DeclarationError):
                factory()
        for factory in (In, Out, State):
            for value in (False, True, None):
                with self.subTest(factory=factory, init=value), self.assertRaises(DeclarationError):
                    kwargs = {"default": 0} if factory is State else {}
                    factory(init=value, **kwargs)

    def test_missing_and_unresolved_annotations_identify_field(self):
        for annotations in ({}, {"value": "UnknownType"}, {"value": "ClassNotDefinedYet"}):
            with self.subTest(annotations=annotations):
                with self.assertRaisesRegex(DeclarationError, "Broken.value"):
                    type("Broken", (Block,), {"__annotations__": annotations, "value": Param()})

    def test_annotation_modes_have_equivalent_schemas(self):
        source = """
from graphion import Array, Block, Control, In, N, Out, Param, State
class Model(Block):
    gain: {q}float{q} = Control(default=1.)
    optional: {q}Model | None{q} = Param(default=None)
    x: {q}Array{q} = In(N)
    y: {q}Array{q} = Out(N)
    count: {q}int{q} = State(default=0)
"""
        # In 3.14 the unquoted form is deferred. In 3.12 it is eager, so
        # self-reference is quoted independently of the other annotations.
        for mode in ("evaluated_or_deferred", "quoted", "postponed"):
            body = source.format(q='"' if mode == "quoted" else "")
            if mode == "postponed":
                body = "from __future__ import annotations\n" + body
            elif mode == "evaluated_or_deferred" and sys.version_info < (3, 14):
                body = body.replace("optional: Model | None", 'optional: "Model | None"')
            with self.subTest(mode=mode), module_with(body) as module:
                schema = module.Model.schema
                self.assertIs(schema["gain"].value_type, float)
                self.assertEqual(schema["optional"].value_type, module.Model | None)
                self.assertEqual(schema["x"].value_type, Array)
                self.assertEqual(schema["x"].axes, (N,))
                self.assertEqual(schema["count"].default, 0)
                model = module.Model()
                self.assertEqual(model.gain, 1.0)

    def test_unrelated_annotations_need_not_resolve(self):
        modes = ["from __future__ import annotations\n"]
        if sys.version_info >= (3, 14):
            modes.append("")
        for prefix in modes:
            source = (
                prefix
                + """
from graphion import Block, Param
class Model(Block):
    ordinary: UnknownType
    nested: UnknownType.Member
    value: int = Param(default=3)
"""
            )
            with self.subTest(prefix=prefix), module_with(source) as module:
                self.assertEqual(module.Model().value, 3)

    def test_inherited_resolution_uses_defining_module_and_class_scope(self):
        with module_with(
            """
from graphion import Array, Block, In, Param
Alias = int
ArrayAlias = Array
class Parent(Block):
    ClassAlias = float
    p: "Alias" = Param(default=1)
    x: "ArrayAlias" = In()
""",
            "_graphion_parent",
        ) as parent:
            with module_with(
                """
from _graphion_parent import Parent
from graphion import Param
Alias = str
class Child(Parent):
    q: "ClassAlias" = Param(default=2.)
""",
                "_graphion_child",
            ) as child:
                self.assertIs(child.Child.schema["p"], parent.Parent.schema["p"])
                self.assertIs(child.Child.schema["p"].value_type, int)
                self.assertIs(child.Child.schema["q"].value_type, float)
                self.assertEqual(child.Child.schema["x"].value_type, Array)

    def test_unquoted_local_annotation_preserves_the_type_object(self):
        with module_with("""
from graphion import Block, Param
def make():
    class LocalType:
        pass
    class Local(Block):
        value: LocalType = Param()
    return Local, LocalType
Model, ValueType = make()
""") as module:
            self.assertIs(module.Model.schema["value"].value_type, module.ValueType)
            self.assertIsInstance(module.Model(value=module.ValueType()).value, module.ValueType)

    def test_quoted_and_postponed_local_names_are_not_reconstructed(self):
        for prefix, annotation in (
            ("", '"LocalType"'),
            ("from __future__ import annotations\n", "LocalType"),
            ("from __future__ import annotations\n", '"LocalType"'),
        ):
            with self.subTest(prefix=prefix, annotation=annotation):
                with self.assertRaisesRegex(DeclarationError, "Local.value"):
                    with module_with(
                        prefix
                        + """
from graphion import Block, Param
def make():
    class LocalType:
        pass
    class Local(Block):
        value: ANNOTATION = Param()
    return Local
make()
""".replace("ANNOTATION", annotation)
                    ):
                        pass

    def test_generic_parameters_resolve_in_all_annotation_modes(self):
        for prefix, annotation in (("", "T"), ("", '"T"'), ("from __future__ import annotations\n", "T")):
            with self.subTest(prefix=prefix, annotation=annotation):
                with module_with(
                    prefix
                    + """
from graphion import Block, Param
class Model[T](Block):
    value: ANNOTATION = Param()
class Child[U](Model):
    other: U = Param()
""".replace("ANNOTATION", annotation)
                ) as module:
                    self.assertIs(module.Model.schema["value"].value_type, module.Model.__type_params__[0])
                    self.assertIs(module.Child.schema["value"], module.Model.schema["value"])
                    self.assertIs(module.Child.schema["other"].value_type, module.Child.__type_params__[0])

    def test_multiple_block_inheritance_is_rejected(self):
        class Other(Block):
            pass

        with self.assertRaisesRegex(DeclarationError, "multiple Block inheritance"):
            type("Combined", (Gain, Other), {})


if __name__ == "__main__":
    unittest.main()
