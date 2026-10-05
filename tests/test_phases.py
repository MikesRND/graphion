"""Phase dependencies, lifecycle failures, and feedback schedules."""

import numpy as np
import pytest

from graphion import (
    Array,
    Block,
    C,
    Control,
    Ctx,
    DeclarationError,
    ExecutionError,
    In,
    N,
    Out,
    ParamError,
    State,
    UnavailablePortError,
    needs_inputs,
)
from graphion.dsp.blocks import FrameDelay, Gain


def configured(block):
    block.setup(Ctx(rate=8, n=3, channels=1))
    return block


def frame(value):
    return np.full((1, 3), value, dtype=np.float64)


class Accumulator(Block):
    x: Array = In(C, N)
    y: Array = Out(C, N)
    held: Array = State(C, N, default=0)

    def output(self):
        np.add(self.held, self.x, out=self.y)

    def update(self):
        np.add(self.held, self.x, out=self.held)


class WrappedDelay(Block):
    x: Array = In(C, N)
    y: Array = Out(C, N)

    def build(self):
        self.delay = self.child("delay", FrameDelay())
        self.connect(self, "x", self.delay, "x")
        self.connect(self.delay, "y", self, "y")

    @needs_inputs(None)
    def output(self):
        self.delay.run_output()

    def update(self):
        self.delay.run_update()


def test_definition_errors_and_inherited_forms():
    with pytest.raises(DeclarationError):

        class Mixed(Gain):
            def process(self):
                pass

    with pytest.raises(DeclarationError):

        class UpdateOnly(Block):
            def update(self):
                pass

    for signature in ("self, missing", "self, y", "self, x=0", "self, *, x", "self, *x", "self, **x"):
        with pytest.raises(DeclarationError):
            exec(
                "class Bad(Block):\n x: Array = In(C, N)\n y: Array = Out(C, N)\n def output(" + signature + "): pass",
                globals(),
                {},
            )

    with pytest.raises(DeclarationError):

        class Static(Block):
            @staticmethod
            def output():
                pass

    container = configured(Block())
    assert container.describe()["blocks"][0]["phases"] == {}
    with pytest.raises(ExecutionError, match="no processing hook"):
        container.run()

    class Inherited(FrameDelay):
        pass

    assert Inherited().describe()["blocks"][0]["phases"] == {"output": [], "update": ["x"]}


@pytest.mark.parametrize("hook", ["process", "output", "update"])
def test_non_none_returns_require_reset(hook):
    methods = {hook: lambda self: 0}
    if hook == "update":
        methods["output"] = lambda self: None
    cls = type("InvalidReturn", (Block,), methods)
    block = configured(cls())
    if hook == "update":
        block.run_output()
    with pytest.raises(ExecutionError, match="return None"):
        (block.run_update if hook == "update" else block.run)()
    for call in (block.run, block.run_output, block.run_update):
        with pytest.raises(ExecutionError, match="reset"):
            call()
    block.reset()
    assert not block._requires_reset
    assert not block._pending


def test_sequence_and_output_only_completion():
    block = configured(FrameDelay())
    block.x = frame(5)
    with pytest.raises(ExecutionError):
        block.run_update()
    result = block.run_output()
    for call in (block.run, block.run_output):
        with pytest.raises(ExecutionError):
            call()
        np.testing.assert_array_equal(result, 0)
        np.testing.assert_array_equal(block.held, 0)
    block.run_update()
    np.testing.assert_array_equal(result, 0)
    np.testing.assert_array_equal(block.held, 5)
    with pytest.raises(ExecutionError):
        block.run_update()
    block.run_output()
    block.reset()
    np.testing.assert_array_equal(block.run_output(), 0)
    block.run_update()

    gain = configured(Gain())
    gain.x = frame(2)
    np.testing.assert_array_equal(gain.run_output(), 2)
    gain.gain = 3
    np.testing.assert_array_equal(gain.run(), 6)
    with pytest.raises(ExecutionError):
        gain.run_update()


def test_shared_input_and_state_advancement():
    block = configured(Accumulator())
    block.x = frame(2)
    result = block.run_output()
    np.testing.assert_array_equal(result, 2)
    np.testing.assert_array_equal(block.held, 0)
    block.run_update()
    np.testing.assert_array_equal(block.held, 2)
    np.testing.assert_array_equal(result, 2)
    block.x = frame(5)
    assert block.run() is result
    np.testing.assert_array_equal(result, 7)
    np.testing.assert_array_equal(block.held, 7)


@pytest.mark.parametrize("hook", ["process", "output", "update"])
def test_hook_and_reset_failures(hook):
    error = ValueError("author failed")

    def fail(self):
        raise error

    def reset(self):
        if self.bad_reset:
            raise error

    methods = {"__annotations__": {"bad_reset": bool}, "bad_reset": Control(default=False), "reset": reset, hook: fail}
    if hook == "update":
        methods["output"] = lambda self: None
    block = configured(type("Failing", (Block,), methods)())
    if hook == "update":
        block.run_output()
    with pytest.raises(ValueError) as caught:
        (block.run_update if hook == "update" else block.run)()
    assert caught.value is error
    block.bad_reset = True
    with pytest.raises(ValueError):
        block.reset()
    with pytest.raises(ExecutionError, match="reset"):
        block.run()
    block.bad_reset = False
    block.reset()
    assert not block._requires_reset
    # Reset failure must also mark a previously healthy block.
    block.bad_reset = True
    with pytest.raises(ValueError):
        block.reset()
    with pytest.raises(ExecutionError, match="reset"):
        block.run()


def test_failure_propagates_through_calls_not_ownership():
    class Failing(Block):
        def process(self):
            raise ValueError("child failed")

    class Parent(Block):
        def build(self):
            self.worker = self.child("worker", Failing())

        def process(self):
            self.worker.run()

    parent = configured(Parent())
    with pytest.raises(ValueError):
        parent.worker.run()
    assert not parent._requires_reset
    parent.reset()
    with pytest.raises(ValueError):
        parent.run()
    assert parent._requires_reset and parent.worker._requires_reset
    parent.reset()
    assert not parent._requires_reset and not parent.worker._requires_reset


@pytest.mark.parametrize("cls", [FrameDelay, WrappedDelay])
def test_update_only_readiness_and_whole_run_precheck(cls):
    block = configured(cls())
    with pytest.raises(UnavailablePortError):
        block.run()
    assert not block._pending and not block._requires_reset
    result = block.run_output()
    np.testing.assert_array_equal(result, 0)
    with pytest.raises(UnavailablePortError):
        block.run_update()
    assert block._pending and not block._requires_reset
    block.x = frame(7)
    block.run_update()
    np.testing.assert_array_equal(result, 0)
    np.testing.assert_array_equal(block.run_output(), 7)
    block.run_update()

    root = Block()
    source = root.child("source", Gain())
    delay = root.child("delay", cls())
    root.connect(source, "y", delay, "x")
    configured(root)
    np.testing.assert_array_equal(delay.run_output(), 0)
    source.x = frame(5)
    source.run()
    delay.run_update()
    np.testing.assert_array_equal(delay.run_output(), 5)


def test_rebinding_reaches_both_phases_and_forwarded_atomic_inputs():
    block = configured(Accumulator())
    for value, total in ((2, 2), (5, 7)):
        block.x = frame(value)
        np.testing.assert_array_equal(block.run(), total)
        np.testing.assert_array_equal(block.held, total)

    class Parent(Block):
        x: Array = In(C, N)
        y: Array = Out(C, N)

        def build(self):
            self.gain = self.child("gain", Gain())
            self.connect(self, "x", self.gain, "x")
            self.connect(self.gain, "y", self, "y")

        def process(self):
            self.gain.run()

    parent = configured(Parent())
    with pytest.raises(UnavailablePortError):
        parent.run()
    for value in (3, 6):
        parent.x = frame(value)
        np.testing.assert_array_equal(parent.run(), value)
    assert parent.describe()["blocks"][0]["phases"] == {"process": ["x"]}


def test_controls_freeze_and_release_without_releasing_an_ancestor():
    class Controlled(Block):
        gain: float = Control(default=1)
        coefficients: Array = Control(default=np.ones(2))

        def output(self):
            with pytest.raises(ParamError):
                self.gain = 2

        def update(self):
            raise ValueError("update failed")

    class Parent(Block):
        def build(self):
            self.worker = self.child("worker", Controlled())

        def output(self):
            self.worker.run_output()

        def update(self):
            self.worker.run_update()

    parent = configured(Parent())
    child = parent.worker
    with pytest.raises(ValueError):
        child.coefficients[:] = 0
    parent.run_output()
    with pytest.raises(ParamError):
        child.gain = 4
    with pytest.raises(ValueError):
        child.run_update()
    with pytest.raises(ParamError):
        child.gain = 4  # The parent's independent pending iteration still freezes it.
    parent.reset()
    child.run_output()
    with pytest.raises(ValueError):
        child.run_update()
    child.gain = 4
    child.coefficients = np.zeros(2)
    assert not child.coefficients.flags.writeable


def test_reset_abandons_the_entire_scope_before_an_early_child_fails():
    class FailingReset(Block):
        fail: bool = Control(default=False)

        def reset(self):
            if self.fail:
                raise ValueError("reset failed")

        def output(self):
            pass

    class Later(FrameDelay):
        gain: float = Control(default=1)

    root = Block()
    first = root.child("first", FailingReset())
    later = root.child("later", Later())
    configured(root)
    later.run_output()
    first.fail = True
    with pytest.raises(ValueError):
        root.reset()
    assert not later._pending
    later.gain = 2
    with pytest.raises(ExecutionError, match="reset"):
        root.run()
    first.fail = False
    root.reset()
    assert not root._requires_reset


def test_feedback_and_exchange_without_a_global_update_barrier():
    class Adder(Block):
        x: Array = In(C, N)
        q: Array = In(C, N)
        y: Array = Out(C, N)

        def output(self):
            self.y[:] = self.x + 0.5 * self.q

    root = Block()
    adder = root.child("adder", Adder())
    delay = root.child("delay", FrameDelay())
    root.connect(delay, "y", adder, "q")
    root.connect(adder, "y", delay, "x")
    configured(root)
    for replay in range(2):
        for value, expected in ((1, 1), (0, 0.5), (0, 0.25), (0, 0.125)):
            adder.x = frame(value)
            delay.run_output()
            np.testing.assert_array_equal(adder.run(), expected)
            delay.run_update()
        root.reset()

    class InitialDelay(FrameDelay):
        initial: float = Control(default=0)

        def reset(self):
            self.held.fill(self.initial)

    for reverse in (False, True):
        root = Block()
        a = root.child("a", InitialDelay(initial=1))
        b = root.child("b", InitialDelay(initial=2))
        root.connect(a, "y", b, "x")
        root.connect(b, "y", a, "x")
        configured(root)
        av, bv = a.run_output(), b.run_output()
        for block in (b, a) if reverse else (a, b):
            block.run_update()
        np.testing.assert_array_equal(av, 1)
        np.testing.assert_array_equal(bv, 2)
        np.testing.assert_array_equal(a.run_output(), 2)
        np.testing.assert_array_equal(b.run_output(), 1)


def test_sample_delay_needs_current_frame():
    class SampleDelay(Block):
        x: Array = In(C, N)
        y: Array = Out(C, N)
        held: Array = State(C, default=0)

        def process(self):
            self.y[:, 0] = self.held
            self.y[:, 1:] = self.x[:, :-1]
            self.held[:] = self.x[:, -1]

    block = configured(SampleDelay())
    with pytest.raises(UnavailablePortError):
        block.run()
    for source, expected in (([1, 2, 3], [0, 1, 2]), ([4, 5, 6], [3, 4, 5])):
        block.x = np.array([source], dtype=float)
        np.testing.assert_array_equal(block.run(), [expected])
    for call in (block.run_output, block.run_update):
        with pytest.raises(ExecutionError):
            call()


def test_input_attributes_are_borrowed_readonly_views():
    class Reader(Block):
        x: Array = In(C, N)

        def output(self):
            assert not self.x.flags.writeable
            self.observed = self.x

    block = configured(Reader())
    with pytest.raises(UnavailablePortError):
        _ = block.x
    source = frame(2)
    block.x = source
    assert not block.x.flags.writeable
    with pytest.raises(ValueError):
        block.x[:] = 0
    block.run()
    assert np.shares_memory(block.observed, source)
    old = block.observed
    block.run()
    assert block.observed is old
    source[:] = 4
    np.testing.assert_array_equal(block.observed, 4)
    assert block.describe()["blocks"][0]["phases"] == {"output": ["x"]}
