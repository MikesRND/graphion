"""Public structural descriptions: schema, lifecycle facts and detached snapshots."""

import gc
import json
import unittest
import weakref
from contextlib import ExitStack
from pathlib import Path
from typing import get_args, get_origin
from unittest.mock import patch

import numpy as np

from graphion import (
    Array,
    Axis,
    Block,
    C,
    Control,
    Ctx,
    In,
    N,
    Out,
    Param,
    State,
    StorageError,
)
from graphion.dsp.blocks import Gain
from graphion.profile.numpy_frame import _storage as arrays
from tests.test_declarations import module_with


def context():
    return Ctx(rate=1_000_000, n=4, channels=2, dtype=np.float64)


def declarations(record):
    return {field["name"]: field for field in record["declarations"]}


def type_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


class ReprForbidden:
    def __repr__(self):
        raise AssertionError("description called repr")


class DescriptionTests(unittest.TestCase):
    def assert_plain_snapshot(self, snapshot):
        def plain(value):
            self.assertIn(type(value), (type(None), bool, int, float, str, list, dict))
            if type(value) is dict:
                for key, item in value.items():
                    self.assertIs(type(key), str)
                    plain(item)
            elif type(value) is list:
                for item in value:
                    plain(item)

        plain(snapshot)
        self.assertEqual(json.loads(json.dumps(snapshot, allow_nan=False)), snapshot)
        self.assertEqual(
            set(snapshot),
            {
                "schema_version",
                "profile",
                "topology",
                "blocks",
                "connections",
                "unresolved",
            },
        )
        self.assertEqual(snapshot["schema_version"], 1)
        self.assertEqual(snapshot["profile"], "python.numpy.frame")
        for block in snapshot["blocks"]:
            self.assertEqual(
                set(block),
                {
                    "path",
                    "definition",
                    "execution_ownership",
                    "phases",
                    "parameters",
                    "controls",
                    "declarations",
                    "children",
                    "context",
                    "storage",
                },
            )
            self.assertEqual(set(block["definition"]), {"module", "qualname"})
            self.assertEqual(block["execution_ownership"], "opaque")
            if block["context"] is not None:
                self.assertEqual(set(block["context"]), {"rate", "rate_exact", "n", "channels", "dtype", "xp"})
            for field in block["declarations"]:
                self.assertEqual(
                    set(field),
                    {
                        "name",
                        "kind",
                        "value_type",
                        "dtype",
                        "shape",
                        "declared_shape",
                        "default",
                    },
                )
                annotation = field["value_type"]
                self.assertEqual(
                    set(annotation),
                    {
                        "array": {"kind"},
                        "named": {"kind", "module", "qualname"},
                        "opaque": {"kind", "type"},
                    }[annotation["kind"]],
                )
                for fact in [field["dtype"], *(field["shape"] or []), *(field["declared_shape"] or [])]:
                    if fact is not None:
                        self.assertEqual(
                            set(fact),
                            {
                                "resolved": {"kind", "value"},
                                "symbol": {"kind", "name"},
                                "axis": {"kind", "name"},
                                "invalid": {"kind", "type"},
                            }[fact["kind"]],
                        )
        for edge in snapshot["connections"]:
            self.assertEqual(set(edge), {"owner", "source", "destination"})
            for endpoint in (edge["source"], edge["destination"]):
                self.assertEqual(set(endpoint), {"path", "port"})
        for fact in snapshot["unresolved"]:
            self.assertEqual(set(fact), {"path", "declaration", "fact", "reason"})
            self.assertIsInstance(fact["reason"], str)
            self.assertTrue(fact["reason"])

    def test_configured_gain_matches_reference_fixture_exactly(self):
        gain = Gain()
        gain.setup(context())
        snapshot = gain.describe()
        fixture = Path(__file__).resolve().parents[1] / "docs/reference/description-example.json"
        self.assertEqual(snapshot, json.loads(fixture.read_text()))
        self.assert_plain_snapshot(snapshot)

    def test_open_description_does_not_evaluate_axes_allocate_or_call_hooks(self):
        class Model(Block):
            x: Array = In(C, N)
            y: Array = Out(Axis("width"))

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
        with patch.object(arrays, "allocate", side_effect=AssertionError("allocation")):
            snapshot = model.describe()
        self.assert_plain_snapshot(snapshot)
        self.assertEqual(snapshot["topology"], "open")
        record = snapshot["blocks"][0]
        self.assertIsNone(record["context"])
        self.assertIsNone(record["storage"])
        fields = declarations(record)
        self.assertEqual(fields["x"]["dtype"], {"kind": "symbol", "name": "Sample"})
        self.assertEqual(
            fields["x"]["shape"],
            [
                {"kind": "symbol", "name": "C"},
                {"kind": "symbol", "name": "N"},
            ],
        )
        self.assertEqual(fields["y"]["shape"], [{"kind": "axis", "name": "width"}])
        facts = [(item["declaration"], item["fact"]) for item in snapshot["unresolved"]]
        self.assertCountEqual(
            facts,
            [
                (None, "topology"),
                (None, "context"),
                (None, "storage"),
                ("x", "dtype"),
                ("x", "shape"),
                ("y", "dtype"),
                ("y", "shape"),
            ],
        )
        self.assertLess(facts.index(("x", "shape")), facts.index(("y", "dtype")))

    def test_literal_and_invalid_selectors_are_described_without_setup_validation(self):
        class Model(Block):
            scalar: Array = In(dtype=int)
            default_dtype: Array = Out(dtype=None)
            malformed_dtype: Array = Out(dtype=[("broken", "f8", -1)])
            literal: Array = Out(np.int64(0), 3, dtype="float32")
            invalid_dtype: Array = Out(2, dtype="not-a-dtype")
            invalid_shape: Array = Out(-1, True, 2.5, dtype=np.float64)

        snapshot = Model().describe()
        fields = declarations(snapshot["blocks"][0])
        self.assertEqual(fields["scalar"]["dtype"], {"kind": "resolved", "value": np.dtype(np.int64).str})
        self.assertEqual(fields["scalar"]["shape"], [])
        self.assertEqual(fields["default_dtype"]["dtype"], {"kind": "resolved", "value": np.dtype(None).str})
        self.assertEqual(fields["malformed_dtype"]["dtype"], {"kind": "invalid", "type": "builtins.list"})
        self.assertEqual(fields["literal"]["dtype"], {"kind": "resolved", "value": np.dtype(np.float32).str})
        self.assertEqual(
            fields["literal"]["shape"],
            [
                {"kind": "resolved", "value": 0},
                {"kind": "resolved", "value": 3},
            ],
        )
        self.assertEqual(fields["invalid_dtype"]["dtype"], {"kind": "invalid", "type": "builtins.str"})
        self.assertEqual(
            fields["invalid_shape"]["shape"],
            [
                {"kind": "invalid", "type": "builtins.int"},
                {"kind": "invalid", "type": "builtins.bool"},
                {"kind": "invalid", "type": "builtins.float"},
            ],
        )
        facts = [(item["declaration"], item["fact"]) for item in snapshot["unresolved"]]
        self.assertIn(("invalid_dtype", "dtype"), facts)
        self.assertIn(("invalid_shape", "shape"), facts)
        self.assertNotIn(("literal", "shape"), facts)
        self.assert_plain_snapshot(snapshot)

    def test_invalid_dtype_does_not_invoke_arbitrary_repr_or_dtype_properties(self):
        class DtypePropertyForbidden:
            @property
            def dtype(self):
                raise AssertionError("description evaluated dtype property")

        for selector in (ReprForbidden(), DtypePropertyForbidden()):

            class Model(Block):
                y: Array = Out(dtype=selector)

            snapshot = Model().describe()
            self.assertEqual(
                declarations(snapshot["blocks"][0])["y"]["dtype"], {"kind": "invalid", "type": type_name(selector)}
            )
            self.assert_plain_snapshot(snapshot)

    def test_current_configuration_is_separate_from_declared_defaults_and_state(self):
        class Model(Block):
            required: int = Param()
            optional: object = Param(default=None)
            gain: float = Control(default=1.0)
            required_control: int = Control()
            count: int = State(default=3)
            memory: Array = State(2, default=np.array([4.0, 5.0]))

        model = Model(required=7, required_control=8)
        model.gain = 2.0
        model.setup(context())
        model.count = 999
        model.memory[:] = 999
        snapshot = model.describe()
        record = snapshot["blocks"][0]
        fields = declarations(record)
        self.assertEqual(
            record["parameters"],
            {
                "required": {"kind": "literal", "value": 7},
                "optional": {"kind": "literal", "value": None},
            },
        )
        self.assertEqual(
            record["controls"],
            {
                "gain": {"kind": "literal", "value": 2.0},
                "required_control": {"kind": "literal", "value": 8},
            },
        )
        self.assertEqual(fields["required"]["default"], {"kind": "required"})
        self.assertEqual(fields["required_control"]["default"], {"kind": "required"})
        self.assertEqual(fields["gain"]["default"], {"kind": "literal", "value": 1.0})
        self.assertEqual(fields["count"]["default"], {"kind": "literal", "value": 3})
        self.assertEqual(fields["memory"]["default"], {"kind": "summary", "dtype": "<f8", "shape": [2]})
        self.assertNotIn("999", json.dumps(snapshot))
        self.assert_plain_snapshot(snapshot)

    def test_python_and_numpy_values_round_trip_without_nonfinite_json_numbers(self):
        class Configuration(Block):
            value: object = Control()

        values = [
            None,
            True,
            123,
            1.25,
            "hello",
            complex(2, -3),
            float("nan"),
            float("inf"),
            float("-inf"),
            complex(float("nan"), float("inf")),
            np.bool_(True),
            np.int64(3),
            np.float32(1.5),
            np.float64(np.inf),
            np.complex128(complex(1, -np.inf)),
            np.str_("numpy"),
        ]
        expected = [
            None,
            True,
            123,
            1.25,
            "hello",
            {"real": 2.0, "imag": -3.0},
            {"float": "nan"},
            {"float": "+inf"},
            {"float": "-inf"},
            {"real": {"float": "nan"}, "imag": {"float": "+inf"}},
            True,
            3,
            1.5,
            {"float": "+inf"},
            {"real": 1.0, "imag": {"float": "-inf"}},
            "numpy",
        ]
        for value, represented in zip(values, expected):
            with self.subTest(type=type(value)):
                snapshot = Configuration(value=value).describe()
                result = {"kind": "literal", "value": represented}
                if isinstance(value, np.generic):
                    result["dtype"] = value.dtype.str
                self.assertEqual(snapshot["blocks"][0]["controls"]["value"], result)
                self.assert_plain_snapshot(snapshot)

    def test_arrays_and_opaque_values_are_summaries_without_repr_or_recursion(self):
        class Configuration(Block):
            value: object = Control()

        bomb = ReprForbidden()
        cyclic = []
        cyclic.append(cyclic)
        for value in [
            bomb,
            [bomb],
            {"value": bomb},
            cyclic,
            lambda: 1,
            np.timedelta64(1, "D"),
            np.datetime64("2020-01-01"),
            np.bytes_(b"bytes"),
            np.void(b"bytes"),
            np.longdouble(1),
            np.clongdouble(1j),
        ]:
            snapshot = Configuration(value=value).describe()
            self.assertEqual(snapshot["blocks"][0]["controls"]["value"], {"kind": "opaque", "type": type_name(value)})
            self.assert_plain_snapshot(snapshot)
        value = np.array([[bomb]], dtype=object)
        snapshot = Configuration(value=value).describe()
        self.assertEqual(
            snapshot["blocks"][0]["controls"]["value"],
            {
                "kind": "summary",
                "dtype": value.dtype.str,
                "shape": [1, 1],
            },
        )
        self.assert_plain_snapshot(snapshot)

    def test_named_array_and_opaque_value_types_use_stored_annotations(self):
        class Model(Block):
            named: np.ndarray = Param(default=np.ones(2))
            array: Array = Control(default=np.zeros(2))
            generic: list[int] = Control(default=[])
            y: Array = Out()

        snapshot = Model().describe()
        fields = declarations(snapshot["blocks"][0])
        self.assertEqual(
            fields["named"]["value_type"],
            {
                "kind": "named",
                "module": "numpy",
                "qualname": "ndarray",
            },
        )
        self.assertEqual(fields["array"]["value_type"], {"kind": "array"})
        self.assertEqual(
            fields["generic"]["value_type"],
            {
                "kind": "opaque",
                "type": type_name(Model.schema["generic"].value_type),
            },
        )
        self.assertIn(
            ("generic", "value_type"), [(item["declaration"], item["fact"]) for item in snapshot["unresolved"]]
        )
        self.assert_plain_snapshot(snapshot)

    def test_typing_alias_representation_does_not_compare_arbitrary_arguments(self):
        class EqualityForbidden:
            def __eq__(self, other):
                raise AssertionError("description compared annotation argument")

            def __repr__(self):
                raise AssertionError("description represented annotation argument")

        arguments = get_args(Array)
        equivalent = get_origin(Array)[arguments]
        opaque = get_origin(Array)[(*arguments[:-1], EqualityForbidden())]

        class Model(Block):
            values: equivalent = Param(default=np.ones(2))
            unusual: opaque = Param(default=np.ones(2))

        snapshot = Model().describe()
        fields = declarations(snapshot["blocks"][0])
        self.assertEqual(fields["values"]["value_type"], {"kind": "array"})
        self.assertEqual(fields["unusual"]["value_type"], {"kind": "opaque", "type": type_name(opaque)})
        self.assert_plain_snapshot(snapshot)

    def test_annotation_modes_and_inherited_types_produce_equivalent_descriptions(self):
        results = []
        for mode in ("evaluated", "quoted", "postponed"):

            def annotation(name):
                return repr(name) if mode == "quoted" else name

            source = "from __future__ import annotations\n" if mode == "postponed" else ""
            source += f"""from graphion import Array, Block, Control, N, Out, Param
Alias = Array
class Parent(Block):
    amount: {annotation("int")} = Param(default=3)
    gain: {annotation("float")} = Control(default=1.0)
class Model(Parent):
    gain: {annotation("float")} = Control(default=2.0)
    y: {annotation("Alias")} = Out(N)
"""
            with module_with(source, "_graphion_description_modes") as module:
                model = module.Model()
                before = model.describe()
                model.setup(context())
                after = model.describe()
                results.append((before, after))
                self.assertEqual(
                    [field["name"] for field in before["blocks"][0]["declarations"]], ["amount", "gain", "y"]
                )
                self.assert_plain_snapshot(before)
                self.assert_plain_snapshot(after)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])

    def test_inherited_annotation_identity_does_not_resolve_in_child_module(self):
        with ExitStack() as stack:
            parent = stack.enter_context(
                module_with(
                    """from graphion import Block, Param
class Value: pass
class Parent(Block):
    value: "Value" = Param(default=None)
""",
                    "_graphion_description_parent",
                )
            )
            child = stack.enter_context(
                module_with(
                    """from _graphion_description_parent import Parent
Value = int
class Child(Parent): pass
""",
                    "_graphion_description_child",
                )
            )
            snapshot = child.Child().describe()
            self.assertIs(child.Child.schema["value"].value_type, parent.Value)
            self.assertEqual(
                declarations(snapshot["blocks"][0])["value"]["value_type"],
                {
                    "kind": "named",
                    "module": "_graphion_description_parent",
                    "qualname": "Value",
                },
            )
        self.assert_plain_snapshot(snapshot)

    def test_local_setup_failure_keeps_context_and_axes_unresolved(self):
        class Failing(Block):
            y: Array = Out(N)

            def setup(self, ctx):
                self.during_setup = self.describe()

            def reset(self):
                raise RuntimeError("reset failed")

        model = Failing()
        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            model.setup(context())
        snapshot = model.describe()
        self.assertEqual(snapshot, model.during_setup)
        self.assertEqual(snapshot["topology"], "open")
        record = snapshot["blocks"][0]
        self.assertIsNone(record["context"])
        self.assertIsNone(record["storage"])
        self.assertEqual(declarations(record)["y"]["shape"], [{"kind": "symbol", "name": "N"}])
        self.assert_plain_snapshot(snapshot)

    def test_child_local_facts_survive_ancestor_setup_failure(self):
        class FailingParent(Block):
            def build(self):
                self.leaf = self.child("leaf", Gain())

            def setup(self, ctx):
                self.leaf.setup(ctx.replace(n=3))
                raise RuntimeError("parent failed")

        model = FailingParent()
        with self.assertRaisesRegex(RuntimeError, "parent failed"):
            model.setup(context())
        snapshot = model.describe()
        parent, child = snapshot["blocks"]
        self.assertEqual(snapshot["topology"], "open")
        self.assertIsNone(parent["context"])
        self.assertEqual(child["context"]["n"], 3)
        self.assertIsNone(child["storage"])
        self.assertEqual(
            declarations(child)["x"]["shape"],
            [
                {"kind": "resolved", "value": 2},
                {"kind": "resolved", "value": 3},
            ],
        )
        subtree = model.leaf.describe()
        self.assertEqual(subtree["topology"], "open")
        self.assertEqual(subtree["blocks"][0]["path"], "root")
        self.assert_plain_snapshot(snapshot)
        self.assert_plain_snapshot(subtree)

    def test_local_success_facts_survive_root_finalization_failure(self):
        class Source(Block):
            y: Array = Out(N)

        class Root(Block):
            y: Array = Out(N)

            def build(self):
                self.a = self.child("a", Source())
                self.b = self.child("b", Source())

        model = Root()
        supplied = np.zeros(4)
        model.a.bind_output("y", supplied)
        model.b.bind_output("y", supplied)
        with self.assertRaises(StorageError):
            model.setup(context())
        snapshot = model.describe()
        self.assertEqual(snapshot["topology"], "open")
        for block in snapshot["blocks"]:
            self.assertEqual(block["context"]["n"], 4)
            self.assertIsNone(block["storage"])
            self.assertEqual(declarations(block)["y"]["shape"], [{"kind": "resolved", "value": 4}])
        self.assert_plain_snapshot(snapshot)

    def test_nested_topology_roles_paths_and_subtree_boundary_records(self):
        class Leaf(Block):
            x: Array = In(N)
            y: Array = Out(N)
            memory: Array = State(N, default=0.0)
            count: int = State(default=0)

        class Branch(Block):
            x: Array = In(N)
            y: Array = Out(N)

            def build(self):
                self.leaf = self.child("leaf/~", Leaf())
                self.connect(self.leaf, "y", self, "y")
                self.connect(self, "x", self.leaf, "x")

        class Root(Block):
            x: Array = In(N)
            y: Array = Out(N)
            direct: Array = Out(N)

            def build(self):
                self.branch = self.child("branch/~", Branch())
                self.unused = self.child("unused", Gain())
                self.connect(self, "x", self.branch, "x")
                self.connect(self.branch, "y", self, "y")

        model = Root()
        model.branch.leaf.bind_output("y", np.zeros(4))
        before = model.describe()
        model.setup(context())
        snapshot = model.describe()
        root_path, branch_path = "root", "root/branch~1~0"
        leaf_path, unused_path = branch_path + "/leaf~1~0", "root/unused"
        self.assertEqual(
            [item["path"] for item in snapshot["blocks"]], [root_path, branch_path, leaf_path, unused_path]
        )
        self.assertEqual(snapshot["blocks"][0]["children"], [branch_path, unused_path])
        self.assertEqual(snapshot["blocks"][1]["children"], [leaf_path])
        self.assertEqual(
            snapshot["blocks"][0]["storage"],
            {
                "x": "external_input",
                "y": "forwarded_output",
                "direct": "direct_output",
            },
        )
        self.assertEqual(
            snapshot["blocks"][1]["storage"],
            {
                "x": "connected_input",
                "y": "forwarded_output",
            },
        )
        self.assertEqual(
            snapshot["blocks"][2]["storage"],
            {
                "x": "connected_input",
                "y": "supplied_output",
                "memory": "array_state",
                "count": "scalar_state",
            },
        )
        expected_edges = []
        for owner, source, source_port, destination, destination_port in [
            (root_path, root_path, "x", branch_path, "x"),
            (root_path, branch_path, "y", root_path, "y"),
            (branch_path, leaf_path, "y", branch_path, "y"),
            (branch_path, branch_path, "x", leaf_path, "x"),
        ]:
            expected_edges.append(
                {
                    "owner": owner,
                    "source": {"path": source, "port": source_port},
                    "destination": {"path": destination, "port": destination_port},
                }
            )
        self.assertEqual(snapshot["connections"], expected_edges)
        self.assertEqual(before["connections"], expected_edges)
        subtree = model.branch.describe()
        self.assertEqual(subtree["topology"], "complete")
        self.assertEqual([item["path"] for item in subtree["blocks"]], ["root", "root/leaf~1~0"])
        self.assertEqual(subtree["blocks"][0]["storage"]["x"], "connected_input")
        self.assertEqual(len(subtree["connections"]), 2)
        self.assertTrue(all(edge["owner"] == "root" for edge in subtree["connections"]))
        self.assertTrue(
            all(
                endpoint["path"] in {"root", "root/leaf~1~0"}
                for edge in subtree["connections"]
                for endpoint in (edge["source"], edge["destination"])
            )
        )
        self.assertEqual(snapshot["unresolved"], [])
        self.assert_plain_snapshot(before)
        self.assert_plain_snapshot(snapshot)
        self.assert_plain_snapshot(subtree)

    def test_snapshot_is_detached_from_live_state_and_survives_collection(self):
        class Model(Block):
            gain: float = Control(default=1.0)
            coefficients: Array = Control(default=np.ones(2))
            y: Array = Out(N)
            memory: Array = State(N, default=0.0)

        model = Model()
        model.setup(context())
        snapshot = model.describe()
        saved = json.dumps(snapshot, allow_nan=False)
        reference = weakref.ref(model)
        array_reference = weakref.ref(model.coefficients)
        output_reference = weakref.ref(model.y)
        model.gain = 5.0
        model.coefficients = np.zeros(3)
        model.y.fill(17)
        model.memory.fill(19)
        self.assertEqual(json.dumps(snapshot, allow_nan=False), saved)
        changed = model.describe()
        self.assertNotEqual(changed["blocks"][0]["controls"], snapshot["blocks"][0]["controls"])
        changed["blocks"][0]["declarations"].clear()
        self.assertEqual(len(model.describe()["blocks"][0]["declarations"]), 4)
        del model
        gc.collect()
        self.assertIsNone(reference())
        self.assertIsNone(array_reference())
        self.assertIsNone(output_reference())
        self.assert_plain_snapshot(snapshot)

    def test_custom_dtype_resolves_only_during_setup_and_cycles_remain_invalid(self):
        class Selector:
            calls = 0

            @property
            def dtype(self):
                self.calls += 1
                return np.dtype(np.float32)

        selector = Selector()

        class Model(Block):
            y: Array = Out(N, dtype=selector)

        model = Model()
        before = model.describe()
        self.assertEqual(selector.calls, 0)
        self.assertEqual(
            declarations(before["blocks"][0])["y"]["dtype"], {"kind": "invalid", "type": type_name(selector)}
        )
        model.setup(context())
        setup_calls = selector.calls
        self.assertGreater(setup_calls, 0)
        after = model.describe()
        self.assertEqual(selector.calls, setup_calls)
        self.assertEqual(
            declarations(after["blocks"][0])["y"]["dtype"], {"kind": "resolved", "value": np.dtype(np.float32).str}
        )
        cyclic_list = []
        cyclic_list.append(cyclic_list)
        cyclic_dict = {}
        cyclic_dict["formats"] = cyclic_dict
        for cyclic in (cyclic_list, cyclic_dict):

            class Cyclic(Block):
                y: Array = Out(dtype=cyclic)

            snapshot = Cyclic().describe()
            self.assertEqual(
                declarations(snapshot["blocks"][0])["y"]["dtype"], {"kind": "invalid", "type": type_name(cyclic)}
            )
            self.assert_plain_snapshot(snapshot)
        self.assert_plain_snapshot(before)
        self.assert_plain_snapshot(after)

    def test_numpy_string_axis_and_connection_names_become_plain_strings(self):
        class Source(Block):
            width: int = Param(default=4)
            y: Array = Out(Axis(np.str_("width")))

        class Destination(Block):
            x: Array = In(N)

        model = Block()
        source = model.child(np.str_("source"), Source())
        destination = model.child(np.str_("destination"), Destination())
        model.connect(source, np.str_("y"), destination, np.str_("x"))
        before = model.describe()
        self.assertEqual(declarations(before["blocks"][1])["y"]["shape"], [{"kind": "axis", "name": "width"}])
        self.assert_plain_snapshot(before)
        model.setup(context())
        self.assert_plain_snapshot(model.describe())


if __name__ == "__main__":
    unittest.main()
