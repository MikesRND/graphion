# Python and NumPy

[Programming reference](index.md)

<a id="python-and-numpy"></a>

## Processing model

This reference defines the concrete Python API and its NumPy storage and
execution contracts. Other Implementation Profiles realize the shared concepts
through their own mechanisms.

`graphion.profile.numpy_frame` implements the Python/NumPy frame
Implementation Profile, identified as `python.numpy.frame` by `describe()`.
It exports `Block`, `Ctx`, declaration types and tokens, and errors.
The root `graphion` package re-exports
those same authoring objects.

DSP library blocks are imported from `graphion.dsp.blocks`:

```python
from graphion.profile.numpy_frame import Block, Ctx
from graphion.dsp.blocks import Gain, FrameDelay, OnePoleLowpass
```

These blocks support this Implementation Profile. Their public library names do
not select or guarantee support for other Implementation Profiles.

The Implementation Profile requires Python 3.12+ and NumPy 2+. Processing is
synchronous: a successful `run()` returns after the numerical work it invokes has
completed. Inputs may then be refilled, and returned outputs may be read
until their Direct Producer begins its next output-producing hook or resets. Calls on the same
tree and writes to its borrowed storage require caller-controlled ordering;
the framework provides no concurrent execution protocol. [Execution and buffer lifetime](execution.md#execution-and-buffer-lifetime)
defines this execution contract.

| Concern | Python contract |
|---|---|
| Authoring | Declare Params, Controls, State, Ins and Outs; use generated construction. |
| Numerical work | Implement `process(self)`, or `output(self)` with optional `update(self)`; read Input attributes and fill Output storage. |
| Storage | Allocate or bind fixed outputs during setup; allocate array State and reusable workspaces during setup. |
| Connections | Record named source and destination ports explicitly. |
| Scheduling | Call `run()`, or split `run_output()` / `run_update()`, in the required order. |
| Results | Return cached read-only views borrowing persistent producer storage. |
| Validation | Check resolved interfaces and declared storage at setup and external input binding. |

Frame shapes are fixed after setup, and every call processes complete
frames. Fixed-size rate conversion uses explicit output dimensions and
child contexts. Output storage has one lifetime contract; output
replacement, per-call destinations and partial-frame processing are outside
this Implementation Profile. Reset restores State. Output-producing hooks compute meaningful initial
Outputs from that State; Reset does not produce Outputs.

Features outside this Implementation Profile are listed in
[the design](../DESIGN.md#outside-this-implementation-profile). Extensions must
preserve the simple block authoring and explicit execution model described here.
