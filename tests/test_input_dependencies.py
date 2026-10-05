"""Explicit Output dependencies and conservative defaults."""

from unittest.mock import Mock

import numpy as np
import pytest

from graphion import Array, Block, DeclarationError, In, N, Out, UnavailablePortError, needs_inputs
from tests.test_phases import configured


class Pair(Block):
    x: Array = In(N)
    z: Array = In(N)
    y: Array = Out(N)

    def output(self):
        np.add(self.x, self.z, out=self.y)


def phases(cls):
    return cls().describe()["blocks"][0]["phases"]


@pytest.mark.parametrize("with_update", [False, True])
def test_undecorated_output_requires_every_input(with_update):
    class Output(Pair):
        pass

    if with_update:

        class Output(Output):
            def update(self):
                pass

    block = configured(Output())
    block.x = np.ones(3)
    with pytest.raises(UnavailablePortError, match="z"):
        block.run_output()
    assert not block._pending and not block._requires_reset
    block.z = np.full(3, 2.0)
    np.testing.assert_array_equal(block.run(), 3)
    assert phases(Output)["output"] == ["x", "z"]


def test_subset_readiness_whole_run_precheck_and_rebinding():
    class Subset(Pair):
        @needs_inputs("x")
        def output(self):
            np.copyto(self.y, self.x)

        def update(self):
            self.seen = self.z.copy()

    block = configured(Subset())
    block.x = np.ones(3)
    with pytest.raises(UnavailablePortError, match="z"):
        block.run()
    assert not block._pending and not block._requires_reset
    np.testing.assert_array_equal(block.run_output(), 1)
    with pytest.raises(UnavailablePortError, match="z"):
        block.run_update()
    block.z = np.full(3, 2.0)
    block.run_update()
    np.testing.assert_array_equal(block.seen, 2)
    block.x = np.full(3, 4.0)
    block.z = np.full(3, 5.0)
    np.testing.assert_array_equal(block.run(), 4)
    np.testing.assert_array_equal(block.seen, 5)
    assert phases(Subset) == {"output": ["x"], "update": ["x", "z"]}


def test_inherited_declarations_and_hook_overrides():
    class Explicit(Pair):
        @needs_inputs("x")
        def output(self):
            np.copyto(self.y, self.x)

    class Inherited(Explicit):
        x: Array = In(N)
        more: Array = In(N)

    class Default(Pair):
        more: Array = In(N)

    class Overridden(Inherited):
        def output(self):
            np.copyto(self.y, self.x)

    class Ordered(Pair):
        @needs_inputs("z", "x")
        def output(self):
            super().output()

    assert phases(Explicit) == phases(Inherited) == {"output": ["x"]}
    assert phases(Default) == phases(Overridden) == {"output": ["x", "z", "more"]}
    assert phases(Pair) == phases(Ordered) == {"output": ["x", "z"]}


@pytest.mark.parametrize("entries", [(), (None, "x"), ("x", "x"), ("missing",), ("y",), (1,), (In(N),)])
def test_invalid_dependencies(entries):
    with pytest.raises(DeclarationError):

        class Invalid(Pair):
            @needs_inputs(*entries)
            def output(self):
                pass


def test_invalid_decorator_uses():
    with pytest.raises(DeclarationError, match="parentheses"):

        class Bare(Block):
            @needs_inputs
            def output(self):
                pass

    with pytest.raises(DeclarationError, match="only one"):

        class Repeated(Block):
            @needs_inputs(None)
            @needs_inputs(None)
            def output(self):
                pass

    for name in ("process", "update", "helper", "reset"):
        with pytest.raises(DeclarationError, match="only"):
            exec(f"class Wrong(Block):\n @needs_inputs(None)\n def {name}(self): pass", globals())
    with pytest.raises(DeclarationError, match="only"):

        class Aliased(Block):
            @needs_inputs(None)
            def output(self):
                pass

            helper = output


def test_none_dependency_and_self_only_signature():
    class Constant(Pair):
        @needs_inputs(None)
        def output(self):
            self.y.fill(3)

    np.testing.assert_array_equal(configured(Constant()).run(), 3)
    assert phases(Constant) == {"output": []}
    for name in ("process", "output", "update"):
        prefix = " def output(self): pass\n" if name == "update" else ""
        with pytest.raises(DeclarationError, match="positional self"):
            exec(f"class Invalid(Block):\n x: Array = In(N)\n{prefix} def {name}(self, x): pass", globals())


def test_unrelated_dynamic_attributes_are_not_decorated_methods():
    class WithMock(Pair):
        observer = Mock()

    block = configured(WithMock())
    block.x = np.ones(3)
    block.z = np.full(3, 2.0)
    np.testing.assert_array_equal(block.run(), 3)
    assert phases(WithMock) == {"output": ["x", "z"]}
