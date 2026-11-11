# graphion

## Introduction

Graphion is a block framework for composing processing models. It provides
declarative interfaces, explicit execution and predictable storage ownership,
with applications in signal processing, machine learning and other data workflows.

The Python implementation uses NumPy and requires Python 3.12+ and NumPy 2+.
The [programming reference](docs/reference/index.md) defines the API; the
[charter](docs/CHARTER.md) and [design](docs/DESIGN.md) explain its principles
and architecture.

Run the [examples](examples/README.md) for live or saved spectra and time traces,
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

`graphion.dsp.blocks` is the DSP library, organized into arithmetic, delay, and
filter modules. Its blocks support the Python/NumPy frame Implementation Profile.
Gain, FrameDelay, and OnePoleLowpass also serve as reference cases for checking
algorithmic behavior across Implementation Profiles. Library names identify
their application; supported Implementation Profiles belong to each
implementation's contract.

## Developer Instructions

### Developer Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
this command from the repository root:

```bash
uv sync --locked
```

uv uses the Python 3.12 version in `.python-version`, creates `.venv`, and
installs Graphion from `src/graphion` in editable mode along with the `dev`
dependency group. `uv.lock` records the resolved dependencies. The development
tools include pytest, mypy, Ruff, Matplotlib, and PySide6 for live plots.
Run the following commands from the repository root; `uv run` selects the
environment automatically.

### Testing

Run pytest:

```bash
uv run --locked pytest
```

### Coding Standards

Format code and apply safe lint fixes:

```bash
uv run --locked ruff check --fix .
uv run --locked ruff format .
```

Check formatting, lint, and types:

```bash
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy
```

Pytest also checks that invalid dimension expressions and dependency decorator
arguments are rejected by mypy.
To run that check on its own:

```bash
uv run --locked python -m tests.check_typing
```

### Distribution Builds

Build the wheel and source distribution:

```bash
uv build
```

### Documentation

Start at the [documentation index](docs/index.md). The
[programming reference](docs/reference/index.md) defines the normative API; the
quick start and consolidated user guide are intentionally brief.

Check documentation links, anchors and Python examples:

```bash
uv run --locked python docs/check.py
```

### Continuous Integration

GitHub Actions runs the formatting, lint, typing, test, documentation, and
distribution-build checks above in one Ubuntu/Python 3.12 job on pushes and
pull requests. Plotting tests use Matplotlib's headless Agg backend.
