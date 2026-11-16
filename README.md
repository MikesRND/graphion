# graphion

## Introduction

Graphion is a block framework for composing processing models. It provides
declarative interfaces, explicit execution and predictable storage ownership,
with applications in signal processing, machine learning and other data workflows.

The Python implementation uses NumPy and requires Python 3.12+ and NumPy 2+.
The [programming reference](docs/reference/index.md) defines the API; the
[charter](docs/CHARTER.md) and [design](docs/DESIGN.md) explain its principles
and architecture.

Run the [examples](examples/README.md) for live or saved spectra, time traces and BER curves,
with shared example block and plotting libraries. The development environment
includes Matplotlib for plotting.

Supported in this release:

- Declarative blocks with Params, Controls, State and In/Out ports, named axes and
  dtypes, and generated constructors that static type checkers understand.
- Explicit `build()`, `setup()`, `reset()` and `run()` with no hidden scheduler.
- Atomic `process` or split `output`/`update` hooks using Input attributes;
  `@needs_inputs(...)` declares narrowed Output dependencies, and
  separate `run_output()` and `run_update()` calls support stateful feedback.
- Framework-allocated, reused output and State buffers; read-only borrowed
  results; caller-supplied outputs with overlap validation.
- Hierarchical composition: child blocks, connections with fan-out, input and
  output forwarding, topology validation at build and tree-wide storage validation at setup.
- Exact rational sample rates through rate-changing stages.
- `describe()` plain-data structural descriptions.

The Python/NumPy frame [Implementation Profile](docs/reference/model.md#implementation-profiles)
is implemented in
`graphion.profile.numpy_frame`.
It combines synchronous, caller-ordered execution with fixed-size NumPy frames.
The root package provides convenient imports for its block-authoring API:

```python
from graphion import Array, Block, Ctx, In, Out
from graphion.dsp.blocks import Gain, FrameDelay, OnePoleLowpass
```

Alternatively, import the same authoring objects from the Implementation Profile's
package:

```python
from graphion.profile.numpy_frame import Array, Block, Ctx, In, Out
```

`graphion.dsp.blocks` provides Gain, FrameDelay, OnePoleLowpass, FirInterpolate,
FirDecimate, Prbs, SymbolMapper, SymbolDemapper, Awgn, SampleDelay, ErrorCounter and
ConstellationViewer.
The [DSP library reference](docs/reference/dsp-library.md) covers these blocks,
`stream_ctx`, FIR design, constellations and energy/SNR helpers. Its blocks
support the Python/NumPy frame Implementation Profile.
Gain, FrameDelay, and OnePoleLowpass also serve as reference cases for checking
algorithmic behavior across Implementation Profiles. Library names identify
their application; supported Implementation Profiles belong to each
implementation's contract.

## Development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run the
development gate from the repository root.

```bash
uv run --locked just check
```

uv selects Python 3.12 from `.python-version` and installs Graphion in editable
mode with the development tools from `uv.lock`. The gate checks formatting, lint,
types, tests, and documentation.

| Command | Purpose |
| --- | --- |
| `uv run --locked just` | List recipes. |
| `uv run --locked just sync` | Install the locked project and development dependencies. |
| `uv run --locked just fix` | Apply safe Ruff fixes and format Python. |
| `uv run --locked just fmt-check` | Check formatting. |
| `uv run --locked just lint` | Run Ruff lint rules. |
| `uv run --locked just types` | Run Pyrefly, including the positive authoring examples. |
| `uv run --locked just test` | Run pytest. Pass paths or pytest flags after `test` to focus the run. |
| `uv run --locked just test tests/test_typing_check.py` | Verify the expected diagnostics for invalid authoring. |
| `uv run --locked just docs` | Check documentation links, anchors, and Python examples. |
| `uv run --locked just build` | Build the wheel and source distribution. |
| `uv run --locked just ci` | Run the development gate and build distributions. |

The [programming reference](docs/reference/index.md) defines the normative API.
Start with the [documentation index](docs/index.md) for the quickstart and guides.

GitHub Actions runs the same checks and distribution build on Ubuntu with
Python 3.12. Plotting tests use Matplotlib's headless Agg backend.
