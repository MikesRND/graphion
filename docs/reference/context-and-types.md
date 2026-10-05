# Context, dtypes and shapes

[Programming reference](index.md)

<a id="context-dtypes-and-shapes"></a>

## Context, dtypes and shapes

`Ctx` belongs to the Python/NumPy frame Implementation Profile and is exported
by both `graphion.profile.numpy_frame` and `graphion`. Its fields configure that
Implementation Profile; they are not universal properties of every block model.
`Ctx` is immutable and provides `replace(**changes)`:

| Field | Meaning |
|---|---|
| `rate` | Positive sample rate in Hz, stored as a Python `int`, `float` or `fractions.Fraction`. |
| `n` | Positive integer samples per frame. |
| `channels` | Positive integer channel count. |
| `dtype` | Numeric NumPy dtype; defaults to `np.float64`. |
| `xp` | Author-facing array namespace; defaults to NumPy. This Implementation Profile supports NumPy only. |

`rate` accepts Python `int`, `float` and `fractions.Fraction` values, plus NumPy
integer and floating scalars. NumPy scalars become Python `int` or `float` values;
conversion of extended-precision NumPy floats may round to Python float precision.
Integer and Fraction rates retain their exact values. Booleans (including NumPy
booleans), strings, complex values and other real-number implementations are
rejected with `SetupError`.

At construction, `float(rate)` must be positive and finite. Overflow or underflow
to zero raises `SetupError`, including for exact rates. This validates the numeric
approximation used by `describe()` without replacing the exact runtime value.
`replace(rate=...)` applies the same rules. The description includes both that
approximation and an exact integer/Fraction record, as specified in
[the description schema](introspection.md#description-schema).

For exact derived rates, start with a Fraction, for example
`Ctx(rate=Fraction(48_000_000, 1001), n=4096, channels=2)` after importing
`Fraction` from `fractions`. Then `ctx.replace(rate=ctx.rate / 4)` retains exact
rational arithmetic. Dividing two integers with `/` produces a float; use a
Fraction for the starting rate or divisor when exactness is required.

`Ctx` validates these basic constraints. Blocks may restrict the supported
dtypes or other values further. For example, a block that calculates
fractional gains can require floating-point or complex samples. Setup
rejects an unsupported `ctx.xp` before the author body runs. `replace()`
preserves `xp` unless explicitly changed; arbitrary extra ctx fields are
not accepted. A namespace field is not a promise of backend portability.

In a port or State declaration:

- A literal dtype such as `np.float32` is used exactly. Python `float`,
  `int`, `complex` and `bool` mean float64, int64, complex128 and bool_.
- `Sample` means this block's `ctx.dtype`. It is not dtype inference or
  an instruction to cast the input.
- `C` and `N` mean this block's channel count and frame length.
- `C` and `N` support multiplication and exact division by integer/Fraction
  constants, fixed Param declarations, or other dimension expressions. Division
  does not truncate. Only the final result must be a non-negative integer.
- An integer axis is a literal size. `Axis("name")` reads a size from a
  declared Param, a class-level property or an instance attribute set during
  setup, including a slot. Ordinary class constants, other descriptors and
  values supplied only by dynamic `__getattr__`/`__getattribute__` lookup are
  unsupported; use a literal dimension or one of the supported sources.
- Array dimensions are in memory-index order. The examples use `(C, N)`.
- A port with no axes carries a zero-dimensional ndarray. This differs
  deliberately from scalar State and scalar configuration.

Port and State specs resolve during setup, after the author's setup body.
They stay fixed for that configured instance. Any unresolved, negative or
nonintegral dimension is a setup error. There is no allocation on access.
Array metadata such as shape and dtype must not be mutated after a buffer
has been configured or supplied as an input.

### Exact dimension expressions

Expressions in port and State declarations resolve during setup using the
instance's context and fixed configuration values:

```python
from fractions import Fraction
from graphion import Array, Block, C, N, Out, Param, State


class Sized(Block):
    ratio: Fraction = Param(default=Fraction(3, 2))
    factor: int = Param(default=2)
    y: Array = Out(C, N * ratio)
    held: Array = State(C, N / factor, default=0)
```

Declare a Param before referencing it in a class body. The expression records
the field reference, not its default value; constructor configuration determines
its value for each instance. `ratio * N` also works. A subclass can write
`N / Sized.factor`; the reference must belong to an ancestor, and a same-name
Param override supplies the instance value. Foreign fields, Controls, State and
one declaration object bound to multiple names are rejected at class definition.

Supported numeric operands are Python and NumPy integers and `Fraction` values.
Booleans and floats are rejected. Literal invalid operands fail at class
definition; invalid configured values and division by zero fail at setup.
Intermediate fractions are retained exactly: `N / 2 * 2` resolves to `N` even
when the reference frame length is odd. Both `N * 3 / 2` and
`N * Fraction(3, 2)` are exact. `N * (3 / 2)` is rejected because Python evaluates
the parenthesized division to a float before constructing the expression.

Arithmetic is limited to multiplication and division on `C`, `N`, and expression
objects. Params themselves have no arithmetic operators; `m / n` on two Param
declarations is unsupported, while `N * m / n` works. `Sample` and named `Axis`
objects have no arithmetic. Use `Axis("name")` for computed sizes and for a bare
Param dimension. Fractions are accepted inside expressions, not as bare axes.

These expressions describe array sizes only. They infer nothing about sample
rates, relative sampling, token consumption/production or execution order.
Expressions are evaluated once at setup, with no per-run traversal.

Axis lookup checks attribute existence without invoking its getter. A missing or
unsupported axis source, including an unset slot, raises `SetupError` naming the
axis and declaration. An existing property's unrelated exceptions, including
`AttributeError`, propagate unchanged; dynamic lookup does not mask that error.
Framework errors for unavailable child configuration retain the axis and child
context as specified by [setup](lifecycle.md#setup).

Every call consumes and produces the full declared frame shapes. Different
ports may have different fixed shapes: a rate-changing block declares its
output size explicitly, and its owner supplies downstream children with an
appropriate ctx. There is no resizing, variable-size publication, or
per-call valid-length argument.

<a id="array-validation"></a>

## Array validation

Signal arrays are numeric NumPy ndarrays; object, structured,
string and non-NumPy arrays are unsupported. Accepted dtype kinds are `b`, `i`,
`u`, `f`, `c`. Validation performs no implicit signal conversion or transfer.
Writability must be known true for supplied outputs; absence of a writable-storage
capability is not success. Read-only views must leave producer/caller array flags
unchanged. Optional metadata diagnostics do not relax required binding checks.
