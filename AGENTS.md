# Working on Graphion

Graphion is a declarative block model for DSP, ML, and AI. The Python/NumPy
frame profile makes execution, state, storage, and model structure explicit.
Keep ordinary blocks simple to author and their numeric contracts reviewable.

## Commands

Run from the repository root with uv installed.

- `uv run --locked just check` checks format, lint, types, tests, and docs.
- `uv run --locked just fix` applies safe Ruff fixes and formats Python.
- `uv run --locked just test tests/test_declarations.py` runs a focused test file.
- `uv run --locked just ci` also builds the wheel and source distribution.

Run `uv run --locked just check` before finishing a change. If it fails, fix
the cause or report the failure.

## Repository map

- `src/graphion/profile/numpy_frame/` implements declarations, lifecycle, and
  storage. `src/graphion/dsp/` contains DSP blocks and helpers.
- `examples/` contains runnable compositions. `tests/` covers runtime contracts
  and the positive and negative typing surfaces.
- Update the normative [programming reference](docs/reference/index.md)
  with API changes. See the [charter](docs/CHARTER.md) and
  [design](docs/DESIGN.md) for project principles and architecture.

## Code and commits

- Use comments to explain why a decision or invariant matters. Document public
  APIs and modules with docstrings.
- Preserve numeric polymorphism, declaration metadata, and runtime acceptance.
  Keep `Sample` and dtype selectors. Resolve diagnostics without numeric
  conversions, concrete port dtypes, or relaxed checks.
- Keep the framework's run paths free of new allocations, casts, or checks.
- Use `misuse(value)` only in rejection tests with an exception type and a
  message fragment. Wrap the individual argument. Wrap the callable only
  when exercising a missing, extra, or otherwise invalid signature.
- Every suppression, `cast`, and `Any` annotation used to quiet the checker
  names its target (a specific diagnostic code or type) and gives a reason.
- Use Conventional Commits with an imperative subject, for example
  `fix(storage): reject overlapping output buffers`.
