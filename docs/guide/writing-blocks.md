# Write a block

A block declares the data and configuration it needs and implements its processing
in `process()`, or `output()` with optional `update()`. Use Params for
fixed configuration, Controls for changes between Block Iterations, ports for
data and State for retained history. Annotations describe values;
field calls describe their role.

Import authoring types from `graphion` or explicitly from
`graphion.profile.numpy_frame`. Reusable DSP blocks, including `Gain`,
`FrameDelay`, and `OnePoleLowpass`, are available from `graphion.dsp.blocks`.
The example below shows how to write a delay block yourself.

```python
import numpy as np
from graphion import Array, Block, In, Out, State, C, N, needs_inputs


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

Each `run()` returns the previous iteration's Input, beginning with zeros.
For feedback, call `run_output()` before its Input is available, then
`run_update()` after the producer runs. The output hook has no Input dependencies;
the update hook requires `x`. Both use fixed, separate Output and State buffers.

Use `process(self)` when one calculation produces Outputs and advances State.
Use `output(self)` alone for a pure function. Hooks must return `None`.
Read Inputs through `self.x`; fill Outputs through `self.y`. By default, Output
Evaluation conservatively requires every Input. Use `@needs_inputs("x", "z")` to
name a subset, or `@needs_inputs(None)` for no Input dependencies. Include Inputs
consumed through Child calls too; the runtime does not verify those reads.

The framework
allocates State and outputs during setup. Reset restores the declared history.
Write arrays in place; assignment does not replace their storage. For a scalar
counter, use `count: int = State(default=0)` and update it by ordinary assignment.

Use `build()` to construct children, `setup(ctx)` for coefficients and reusable
scratch, and `reset()` for custom initial conditions. Declared arrays become
available after the author setup body. Initialize State in reset; processing
must fully define Outputs. See the [author guidance](../model-semantics.md#author-guidance).

Details: [declarations](../reference/declarations.md),
[context and dtypes](../reference/context-and-types.md),
[lifecycle](../reference/lifecycle.md). Next: [compose a model](composing-models.md).
