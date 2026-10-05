"""Static typing surface checked by ordinary mypy; not collected as a test module.

The check functions are annotated so mypy checks their bodies without
`check_untyped_defs`. They are never called.
"""

from fractions import Fraction
from typing import assert_type

import numpy as np

from graphion import Array, Axis, Block, C, Control, Ctx, In, N, Out, Param, State
from graphion.dsp.blocks import FrameDelay, Gain, OnePoleLowpass
from graphion.profile import numpy_frame as profile


class SizedBase(Block):
    factor: int = Param()
    x: Array = In(C, N)
    y: Array = Out(C, N / factor)


class SizedChild(SizedBase):
    factor: int = Param(default=2)
    y: Array = Out(C, N / SizedBase.factor)


class FractionSized(Block):
    ratio: Fraction = Param(default=Fraction(3, 2))
    numerator: int = Param(default=3)
    denominator: int = Param(default=2)
    y: Array = Out(C * 2, N * ratio)
    z: Array = Out(C, N / denominator * numerator)
    state: Array = State(C, ratio * N, default=0)


def dimension_checks() -> None:
    base = SizedBase(factor=2)
    child = SizedChild()
    fraction = FractionSized(ratio=Fraction(4, 3))
    assert_type(base.factor, int)
    assert_type(child.factor, int)
    assert_type(fraction.ratio, Fraction)
    assert_type(base.port_spec("y"), profile.PortSpec)
    assert_type(base.port_spec("y").shape, tuple[int, ...])


class TypedGain(Block):
    gain: float = Control(default=1.0)
    x: Array = In(C, N)
    y: Array = Out(C, N)

    def process(self):
        np.multiply(self.x, self.gain, out=self.y)


class TwiceGain(TypedGain):
    gain: float = Control(default=2.0)


class RequiredGain(TypedGain):
    gain: float = Control()


class ConfiguredGain(TypedGain):
    factor: int = Param()
    count: int = State(default=0)

    def process(self):
        super().process()
        self.count += 1


class NamedWidth(Block):
    width: int = Param(default=2)
    x: Array = In(C, Axis("width"))
    y: Array = Out(C, Axis("width"))

    def process(self):
        np.copyto(self.y, self.x)


class TypedDelay(Block):
    x: Array = In(C, N)
    y: Array = Out(C, N)
    held: Array = State(C, N, default=0.0)

    def process(self):
        np.copyto(self.y, self.held)
        np.copyto(self.held, self.x)


class SetupSized(Block):
    factor: int = Param(default=2)
    x: Array = In(C, N)
    y: Array = Out(C, Axis("width"))

    def setup(self, ctx: Ctx):
        self.width = ctx.n // self.factor

    def process(self):
        np.copyto(self.y, self.x[:, :: self.factor])


def field_checks() -> None:
    gain = TypedGain()
    TypedGain(gain=2.0)
    TwiceGain()
    RequiredGain(gain=2.0)
    configured = ConfiguredGain(factor=2)
    NamedWidth(width=3)
    TypedDelay()
    assert_type(gain.gain, float)
    assert_type(gain.y, Array)
    assert_type(configured.factor, int)
    assert_type(configured.count, int)
    gain.gain = 3.0
    gain.x = np.ones((2, 4))


def setup_axis_checks() -> None:
    SetupSized()


class ProfileGain(profile.Block):
    gain: float = profile.Control(default=1.0)
    x: profile.Array = profile.In(profile.C, profile.N)
    y: profile.Array = profile.Out(profile.C, profile.N)

    def process(self):
        np.multiply(self.x, self.gain, out=self.y)


def profile_and_library_checks() -> None:
    custom = ProfileGain(gain=2.0)
    gain = Gain(gain=2.0)
    delay = FrameDelay()
    lowpass = OnePoleLowpass(pole=0.5)
    assert_type(custom.gain, float)
    assert_type(custom.y, Array)
    assert_type(gain.gain, float)
    assert_type(gain.y, profile.Array)
    assert_type(delay.held, Array)
    assert_type(lowpass.pole, float)
    block: Block = gain
    explicit: profile.Block = block
    explicit.setup(profile.Ctx(rate=48_000, n=4, channels=1))


class TypedDependencies(Block):
    x: Array = In(N)
    y: Array = Out(N)

    @profile.needs_inputs("x")
    def output(self) -> None:
        np.copyto(self.y, self.x)


class TypedConstant(TypedDependencies):
    @profile.needs_inputs(None)
    def output(self) -> None:
        self.y.fill(0)


def dependency_typing_checks() -> None:
    from collections.abc import Callable

    explicit: Callable[[], None] = TypedDependencies().output
    independent: Callable[[], None] = TypedConstant().output
    assert_type(explicit(), None)
    assert_type(independent(), None)
