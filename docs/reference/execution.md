# Execution

[Programming reference](index.md)

<a id="execution-and-buffer-lifetime"></a>

## Execution and buffer lifetime

Runnable Blocks choose one authoring form. The shared obligations are in
[Model Semantics](../model-semantics.md); this page defines the NumPy calling contract,
*caller-ordered synchronous execution*.

| Form | Author hooks | Framework entry points |
| --- | --- | --- |
| Atomic | `process(self)` | `run()` |
| Split with State Update | `output(self)`, `update(self)` | `run()`, or `run_output()` then `run_update()` |
| Output only | `output(self)` | `run()` or `run_output()` |

Hooks take only positional `self`, read borrowed read-only Inputs through
attributes such as `self.x`, and fill Output storage through `self.y`. Extra
parameters, defaults, variadics, static methods, and class methods are rejected
at class definition. Mixing `process` with either split hook, or defining `update`
without `output`, is invalid, including inherited hooks. Declaration-only classes
may omit hooks; `run()` on such an instance raises `ExecutionError`.

### Input dependencies

By default, every hook conservatively requires all declared Inputs, including
forwarded Inputs. Only `output()` dependencies can be narrowed:

| Declaration on `output()` | Required Inputs |
| --- | --- |
| No decorator | All declared Inputs, with or without `update()`. |
| `@needs_inputs(None)` | None. |
| `@needs_inputs("x", "z")` | Only `x` and `z`. |

Import `needs_inputs` from `graphion` or `graphion.profile.numpy_frame`. Entries
are strings naming Inputs. An empty call, bare decorator, mixed `None` and names,
unknown or non-Input names, duplicate names, unsupported values, repeated decorators,
and use on another method raise `DeclarationError` no later than class definition.
Inherited decorated hooks keep their named dependencies; inherited undecorated
hooks require all subclass Inputs. Overriding a hook without a decorator restores
the conservative default. Metadata is resolved per class in declaration order.

To break a feedback loop, exclude Inputs from that loop. A delay uses
`@needs_inputs(None)` to evaluate its held State before `update()` captures the
current Input. A narrowed Composite declaration must include boundary Inputs
consumed through Child calls as well as direct reads. Connections remain the
binding authority; the decorator does not pass values or schedule Children.

Dependency completeness is an author contract. Input attributes remain readable
regardless of Phase; reading an excluded, connected Input can silently observe a
stale value. There is no runtime read tracking or AST verification. Every Input
actually read, directly or through Child operations, must be included.

### Entry points

`run()` checks availability for the whole iteration before invoking any hook.
It calls `process` once, or `output` followed by `update` when defined.
`run_output()` checks only output dependencies and leaves an iteration pending when
an update hook exists. `run_update()` requires that pending iteration and checks its
own dependencies. A second output-producing call while pending, an update without
pending output, or a split entry point on an Atomic Block raises `ExecutionError`.
An output-only evaluation completes its iteration immediately.

Processing requires successful setup of the entire lifecycle tree. An unassigned
required external Input raises `UnavailablePortError` naming the consuming Block
and missing Inputs. Connected storage passes this check even before its producer
has run: availability checking does not track freshness. The caller must supply a
valid Schedule and meaningful values. Bound Inputs are not consumed automatically.

`run()` and `run_output()` return the cached read-only Output view, a cached tuple
in declaration order for multiple Outputs, or `None` for a sink. `run_update()`
returns `None`. The [storage contract](storage-and-lifetime.md#the-buffer-contract)
defines result lifetime. All processing hooks must return `None`; any other return
counts as a hook failure.

### Controls and failures

Controls on a Split Block and its descendants freeze immediately before its
output hook is invoked. Assignment raises `ParamError` until the iteration completes
or is abandoned. A pending ancestor independently freezes descendant Controls.
Array Controls are read-only; arbitrary mutable configuration remains an author
responsibility.

Any invoked hook failure abandons that Block's iteration and marks it as requiring
Reset. Every processing entry point then raises `ExecutionError` until Reset
succeeds. Original hook exceptions propagate; invalid returns raise `ExecutionError`.
There is no rollback of partial writes or effects. Each enclosing processing call
from which the exception escapes also fails; an idle ownership ancestor is not
marked. Enclosing hooks must not suppress a Child's processing failure and report
success. Reset the outermost Block whose invocation failed.

Rejections before hook invocation leave the called Block unchanged and are retryable.
A failed hook releases its own Control freeze, but not an independently pending
ancestor's freeze. [Reset](lifecycle.md#reset) abandons pending work throughout its
scope before restoring State.

Purity, complete Outputs, matching shared Input values between Phases, and the ban
on invoking Reset from processing are author/caller contracts. The runtime does
not snapshot arrays, compare values, or intercept individual writes.

### Caller-written schedules

A host schedules roots; Composites schedule Children in ordinary Python. Calls may
be conditional or repeated across Block Iterations. Connection records determine
bindings, not order. No automatic scheduler, global iteration or commit barrier is
implied. Use explicit State and phased calls for
[feedback](examples.md#feedback-across-blocks).

Input views, phase readiness, and result objects are cached during setup;
external rebinding refreshes affected Input views and readiness. Binding checks shape, dtype and overlap before changing readers. Refilling
a retained host buffer requires no rebinding. Array metadata must remain fixed;
the normal processing path performs no shape, dtype or overlap scans and creates
no export views or argument collections.

<a id="allocating-and-reusing-storage-use-the-same-interface"></a>

### Allocating and reusing storage use the same interface

Both of these are supported block bodies:

```python
# Convenient expression: allocates a temporary result, then copies it.
def process(self):
    self.y[...] = self.x * self.gain
```

```python
# Reuse the output directly: avoids a new result array for this multiply.
def process(self):
    np.multiply(self.x, self.gain, out=self.y)
```

Optimization changes the numerical expression, not the declarations,
connections or scheduling. `self.y[...] = expression` does not eliminate
allocations inside the expression. NumPy ufunc `out=` parameters can write
directly to a destination; some operations or overlapping operands can
still need internal temporaries.

For larger expressions, allocate scratch in `setup()` and use operations
with explicit destinations. Private scratch is not reset unless the author
needs that behavior. Libraries that return newly allocated results remain
usable by copying those results into the output buffer:

```python
# result_returning_kernel is a placeholder for an allocating library call.
def process(self):
    result = result_returning_kernel(self.x)
    np.copyto(self.y, result, casting="no")
```

This path has both a temporary result and a copy. It is supported, but it
does not meet the no-new-data-buffer target. Use a kernel with an explicit
destination when that matters, and measure its internal workspace needs.
Every directly produced output keeps the same storage and borrowing
contract regardless of the numerical implementation used to fill it.

<a id="allocation-free-processing"></a>

### Allocation-free processing is an explicit engineering target

The target is **no per-run allocation of signal, State or workspace
data buffers** for blocks written to use reusable storage. The framework
must not introduce mandatory result arrays, frame copies or buffers at
composite boundaries. A model's framework-owned persistent array storage
can therefore be accounted for after setup, rather than growing with the
number of frames processed. Author-created temporaries and retained
snapshots are additional memory and remain the author's choice.

The normal `run()` path should also avoid per-call input dictionaries,
output lists/tuples and new bookkeeping collections. Cache execution
metadata and the return object during setup. An allocating convenience
adapter can be added separately; it must not be required on this path.

This is not a claim that arbitrary Python/NumPy code performs zero heap
allocations. Array views can create small Python objects, scalar updates
can allocate, and numerical libraries may allocate internally. A stricter
allocation-free implementation needs measurement and backend-specific
work. The interface must accommodate that implementation without changing
block wiring or requiring a compiler for the model. This Implementation Profile
requires backends that provide stable, mutable destination storage
and destination-writing operations. It does not promise allocation-free
processing for arbitrary Array API implementations.

No silent buffer pooling, liveness inference, allocation detector or
automatic memory planner is required. These are straightforward buffer
ownership rules, with an optimization path visible in ordinary block code.
