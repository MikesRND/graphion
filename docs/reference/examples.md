# Worked examples

[Programming reference](index.md)

These examples show complete block definitions and explain their numerical
behavior, storage use and execution order. They cover stateful filters,
composition, rate conversion and feedback.

<a id="basic-authoring-and-use"></a>
## Basic authoring and use

```python
from numbers import Integral

import numpy as np
from graphion import (
    Array,
    Block,
    Control,
    Ctx,
    In,
    Out,
    Param,
    State,
    Sample,
    C,
    N,
    SetupError,
    needs_inputs,
)


class Gain(Block):
    gain: float = Control(default=1.0)
    x: Array = In(C, N)
    y: Array = Out(C, N)

    def output(self):
        np.multiply(self.x, self.gain, out=self.y)


amp = Gain(gain=0.5)  # generated constructor
amp.setup(Ctx(rate=20_000_000, n=4096, channels=2, dtype=np.complex64))

frame = np.ones((2, 4096), dtype=np.complex64)
amp.x = frame
y = amp.run()  # cached read-only result view
assert y is not amp.y  # amp.y is the writable producer array
assert np.shares_memory(y, amp.y)
assert not y.flags.writeable
saved = y.copy()  # an explicit retained snapshot

amp.gain = 0.25  # an ordinary control update
another_frame = np.full((2, 4096), 4, dtype=np.complex64)
amp.x = another_frame
assert amp.run() is y  # same view, updated output contents
```

`Sample`, `C` and `N` resolve to `ctx.dtype`, `ctx.channels` and `ctx.n`.
The framework allocates directly produced Outs and array State from these
declarations during setup. Inputs borrow external or connected storage.

The examples below use floating-point or complex sample types. There is
no implicit dtype conversion of signal arrays.

<a id="examples"></a>
## Examples

The examples use the imports and Gain definition in [basic authoring and use](#basic-authoring-and-use). Sample
dtypes are floating-point or complex.

<a id="a-stateful-filter"></a>
### A stateful filter

```python
class OnePoleLowpass(Block):
    pole: float = Param(default=0.9)
    x: Array = In(C, N)
    y: Array = Out(C, N)
    memory: Array = State(C, default=0.0)

    def setup(self, ctx):
        if ctx.dtype.kind not in "fc":
            raise SetupError("OnePoleLowpass requires floating or complex samples")
        if not 0.0 <= self.pole < 1.0:
            raise SetupError("pole must be in [0, 1)")
        self._feed = 1.0 - self.pole
        self._work = np.empty(ctx.channels, dtype=ctx.dtype)

    def process(self):
        x, y = self.x, self.y
        memory, work = self.memory, self._work
        for i in range(x.shape[1]):
            np.multiply(memory, self.pole, out=memory)
            np.multiply(x[:, i], self._feed, out=work)
            np.add(memory, work, out=memory)
            y[:, i] = memory
```

This is a reference recurrence for state tests, not a high-throughput RF
filter kernel. It reuses its data buffers, but slicing creates view
objects and the per-sample Python loop has overhead. It needs no author
constructor or reset method. Array-valued coefficients in a more general
filter can be declared as `coefficients: Array = Param()`.

<a id="a-connected-composite"></a>
### A connected composite

```python
class Chain(Block):
    first_pole: float = Param(default=0.8)
    second_pole: float = Param(default=0.9)
    x: Array = In(C, N)
    y: Array = Out(C, N)

    def build(self):
        self.a = self.child("a", OnePoleLowpass(pole=self.first_pole))
        self.b = self.child("b", OnePoleLowpass(pole=self.second_pole))
        self.connect(self, "x", self.a, "x")
        self.connect(self.a, "y", self.b, "x")
        self.connect(self.b, "y", self, "y")

    def process(self):
        self.a.run()
        self.b.run()
```

The inherited setup handles both children. `Chain.y` refers to `b.y`'s
storage without allocating another frame buffer. The three connections
used by the runtime are exactly the three connections a diagram exporter
sees. Calling either the parent or `b` can overwrite that borrowed output.

<a id="rate-conversion-and-a-child-context"></a>
### Rate conversion and a child context

```python
class Downsample(Block):
    factor: int = Param(default=4)
    x: Array = In(C, N)
    y: Array = Out(C, N / factor)

    def setup(self, ctx):
        if isinstance(self.factor, bool) or not isinstance(self.factor, Integral) or self.factor <= 0:
            raise SetupError("factor must be a positive integer")

    def output(self):
        np.copyto(self.y, self.x[:, :: self.factor], casting="no")


class ReducedRate(Block):
    factor: int = Param(default=4)
    x: Array = In(C, N)
    y: Array = Out(C, N / factor)

    def build(self):
        self.down = self.child("down", Downsample(factor=self.factor))
        self.post = self.child("post", Gain(gain=0.5))
        self.connect(self, "x", self.down, "x")
        self.connect(self.down, "y", self.post, "x")
        self.connect(self.post, "y", self, "y")

    def setup(self, ctx):
        self.down.setup(ctx)  # validates the factor first
        self.post.setup(ctx.replace(n=self.down.port_spec("y").shape[1], rate=ctx.rate / self.factor))

    def process(self):
        self.down.run()
        self.post.run()
```

`ReducedRate` preserves exact rates when `ctx.rate` is a `Fraction`; an integer
rate divided by its integer factor produces a float. See [rate arithmetic](context-and-types.md#context-dtypes-and-shapes).

`N / factor` resolves the frame size exactly and rejects a nonintegral result.
The parent reads the child's resolved size with `port_spec("y").shape[1]`.
The sample-rate calculation remains explicit in the parent; dimension
expressions do not infer it.

Downsample only selects samples. It assumes the signal is already
appropriately band-limited; any anti-alias filtering belongs before it.
The slice is a view, and `copyto` writes its data into the existing output.
The code does not allocate another output data buffer, although creating
the view itself creates an array object.

<a id="feedback-with-explicit-state"></a>
### Feedback with explicit State

```python
class Feedback(Block):
    gain: float = Control(default=0.5)
    x: Array = In(C, N)
    y: Array = Out(C, N)
    previous: Array = State(C, N, default=0.0)

    def process(self):
        y = self.y
        np.multiply(self.previous, self.gain, out=y)
        np.add(self.x, y, out=y)
        np.copyto(self.previous, y, casting="no")
```

This is one frame of feedback history, owned by Feedback. History and
output are distinct preallocated arrays. Reset restores the history; the
next execution overwrites the output. For each frame index k, the
recurrence is `y[k] = x[k] + gain * y[k-1]`, with zero initial history.
At gain 0.5, an impulse frame followed by zero frames gives weights
`1, 0.5, 0.25, 0.125, ...`. There are no full-frame arithmetic temporaries
in this expression sequence.

<a id="a-frame-delay"></a>
### A frame delay

```python
class FrameDelay(Block):
    x: Array = In(C, N)
    y: Array = Out(C, N)
    held: Array = State(C, N, default=0.0)

    @needs_inputs(None)
    def output(self):
        np.copyto(self.y, self.held, casting="no")

    def update(self):
        np.copyto(self.held, self.x, casting="no")
```

Its Output is the Input captured by the previous completed Block Iteration.
`run()` performs both Phases; separate phase calls permit feedback scheduling.
Two copies preserve private State and stable Output storage. For two
delays connected in series and each run once per iteration in upstream
order, the total delay is two iterations. There is no global commit phase.

An owner may process a subtree more than once; its state and delays then
advance more than once. `FrameDelay` is not a guarantee about an arbitrary
feedback graph's timing. In a cyclic manual schedule, the points at which
values are read and state is updated still matter. The Feedback example
shows the simplest way to make the intended feedback recurrence explicit.

<a id="feedback-across-blocks"></a>

## Feedback across blocks

Use a Split delay to supply feedback before its next Input is available.
The Connection records describe the cycle; the Composite supplies its Schedule:

```python
class Adder(Block):
    x: Array = In(C, N)
    q: Array = In(C, N)
    y: Array = Out(C, N)

    def output(self):
        np.multiply(self.q, 0.5, out=self.y)
        np.add(self.x, self.y, out=self.y)


class FeedbackLoop(Block):
    x: Array = In(C, N)
    y: Array = Out(C, N)

    def build(self):
        self.delay = self.child("delay", FrameDelay())
        self.adder = self.child("adder", Adder())
        self.connect(self, "x", self.adder, "x")
        self.connect(self.delay, "y", self.adder, "q")
        self.connect(self.adder, "y", self.delay, "x")
        self.connect(self.adder, "y", self, "y")

    def process(self):
        self.delay.run_output()
        self.adder.run()
        self.delay.run_update()
```

With delay State initially zero and constant-frame Inputs `1, 0, 0, 0`,
Outputs are `1, 0.5, 0.25, 0.125`. Reset restores the initial State and replay
reproduces those results. Output buffers supply no history; the delay's State does.

Two delays can also exchange frames. Starting with State `1` and `2`, evaluate
both Outputs, then update both in either order. Current Outputs remain `1, 2`;
next iteration they become `2, 1`. Each update copies into private State without
changing the frame the other delay reads. No global update barrier is needed:
only the dependencies and Borrow lifetimes constrain the Schedule.

A one-sample delay operating on whole frames still needs current frame data and
cannot break this frame-level cycle. See [acceptance examples](../model-semantics.md#acceptance-examples)
for delay, series, exchange, wrapped-delay, and granularity cases.

<a id="host-access"></a>
### Host access

```python
fx = Chain(first_pole=0.8, second_pole=0.9)
fx.setup(Ctx(rate=20_000_000, n=4096, channels=2, dtype=np.complex64))

for frame in stream:
    fx.x = frame
    sink(fx.run())  # sink consumes before next run

# Input attributes expose the current borrowed read-only view.
# Output Borrows remain valid until their Direct Producer runs again or resets.
latest_input = fx.x
latest_output = fx.y
intermediate = fx.a.y
snapshot = fx.y.copy()  # retain past the next run
```

This convenient pattern validates each different input array and creates
its read-only view at binding. If the source can fill a caller-owned buffer
directly, bind once instead:

```python
input_buffer = np.empty((fx.ctx.channels, fx.ctx.n), dtype=fx.ctx.dtype)
fx.x = input_buffer
while source.read_into(input_buffer):
    sink(fx.run())
```

Here `source.read_into` represents an external I/O API that fills a complete
frame and returns whether one is available. It is not a framework method.
Input binding retains the current caller array and its view; the framework
does not cache every distinct frame by identity.
Driver-ring registration and bounded binding caches are outside this API.

The Python model requires no compilation step. An author may call a
numerical helper or a compiled kernel from `process()`. A callable adapter
can assign the external input and call `run()`; an adapter promising
independent retained results must also copy the output explicitly. Such
adapters reuse the same block implementation and declare their own result
lifetime contract.
