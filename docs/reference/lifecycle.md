# Lifecycle

[Programming reference](index.md)

<a id="construction-children-setup-and-reset"></a>
## Construction, children, setup and reset

<a id="children"></a>

### Children

```python
def build(self):
    self.filter = self.child("filter", OnePoleLowpass(pole=0.8))
```

`child(name, block)` registers a lifecycle/diagram child and returns it.
The registration name is its stable name in the hierarchy. A child can
also be supplied as a constructor Param and registered in `build()`;
it does not have to be created there.

Registration requires a previously unconfigured child and at most one lifecycle
parent. Names must be nonempty strings, unique within the parent, and must not
equal any of the parent's declaration names. They need not be Python identifiers;
description paths use the [documented escaping](introspection.md#description-schema).
Invalid names, declaration-name collisions, duplicate names, invalid child state,
a second lifecycle parent and registration cycles raise `RegistrationError` at
registration. This checks recursive setup/reset bookkeeping, not signal-flow
feedback.

Registration is allowed before setup or in the parent's setup body.
It closes when the parent finishes setup. Prefer `build()` where the
structure does not depend on ctx. Arbitrary other Python references
between blocks are allowed and carry no lifecycle meaning.

<a id="setup"></a>

### Setup

`setup(ctx)` is a framework operation around an optional author body:

1. Validate the supported namespace, store ctx and enter setup. A second
   setup attempt raises. A registered child is configured only within its
   parent's setup; only the lifecycle root performs tree finalization.
2. Run the author's `setup(ctx)` body. It may compute coefficients and
   axis sizes, register/wire children, and explicitly set up children
   with derived contexts.
3. Resolve this block's declarations. Allocate its array State and each
   directly produced Out that has no supplied buffer. Validate supplied
   buffers instead of allocating replacements. Forwarded Outs allocate
   no storage of their own.
4. Set up each remaining registered child with this block's ctx, in
   registration order. Registered children are configured as part of
   their parent's setup, not independently beforehand.
5. Validate this owner's connections whose specs are available. Create direct
   producer export views; resolve local forwarded output exports and cache this
   block's result object. Leave input bindings dependent on ancestor or sibling
   wiring provisional until root finalization (see [whole-tree finalization](#whole-tree-finalization)).
6. Restore this block's State defaults, then run its author `reset()` body,
   with children already initialized. Do not reset those children a second time.
7. Finish local configuration and close this block's registration/wiring.
   An explicitly configured child can supply configured properties and resolved
   [port specs](introspection.md#resolved-port-specs) to its parent,
   but public processing remains unavailable until the whole tree is ready.
8. At the root, finalize every connection-dependent input binding and input
   readiness record from the final endpoint records. Validate all connections,
   then build the complete writer list and check supplied storage and external
   inputs. Mark the tree ready only after all checks succeed. No child's `run()`
   bypasses pending or failed finalization.

Reading `block.ctx` before setup has assigned its context raises `SetupError`
identifying that block. The context is available to the author setup body and
remains available during reset and execution. It is managed by the framework;
assigning to `block.ctx` raises `SetupError`. Pass the context to `setup()` instead.

The author setup body computes sizes, coefficients and private scratch
from configuration and ctx. Automatically allocated Out/State arrays are
available after that body, during reset and execution. To supply custom
Out storage, allocate it and call `bind_output()` in the setup body;
the framework will not also allocate a default buffer.

Initial conditions that need declared storage go in the reset body. A
child with a different ctx is set up explicitly before step 4. The same
rule applies when a parent's dimensions read a child's configured
properties, even if both use the same ctx. Reading only construction-time
child Params does not require setup. If an axis reads unavailable child
configuration, the `SetupError` identifies the unresolved axis and child,
preserving the underlying cause. Unrelated exceptions raised by author axis
properties propagate unchanged rather than being relabeled as missing child setup.
There is no lazy allocation or setup-time execution of the numerical graph.

Setup runs once. A setup failure makes that constructed tree unusable;
fix the configuration and construct a new tree. Recovery from partial
setup is unsupported.

<a id="reset"></a>

### Reset

Public `reset()` requires successful setup of the entire lifecycle tree.
The initial State restoration and author reset bodies invoked internally
by setup run before that readiness flag is set. A public reset:

1. Abandons all pending iterations in the entire subtree before any reset body runs.
2. Recursively resets registered children in registration order.
3. Restores this block's declared State defaults in their existing storage.
4. Runs this block's author `reset()` body.

Author reset bodies should restore only their own block's state. Children have
already been initialized or reset by the framework; calling `child.reset()` from
the parent's setup-time reset body raises `SetupError` because the tree is not
yet ready.

Params, Controls, external input values, output storage and connection
records remain. The framework does not clear or replace output buffers.
Their contents are unspecified after reset. Reset bodies must not write Outputs. Array State and outputs
retain their identities, as do cached export views and return tuples.
Ordinary scratch is left to the author's hook. The writer list is unchanged.

Calling `child.reset()` directly resets only that child's subtree once the
tree is ready. Parent and sibling state remain unchanged. Use the root's
reset for a complete model restart. Registering an already-built subtree
must give all its descendants the same root readiness state. Readiness
storage and propagation are private implementation details.

A retained result requires an earlier explicit copy. Reset is not a
snapshot operation and provides no promise about the data in borrowed
result references after it returns.

A failed Reset makes the called Block require Reset, even if previously healthy.
Processing remains blocked until Reset succeeds. Reset does not roll back external
effects. Controls may be corrected after failure unless an ancestor remains pending.

Processing hooks must not call `reset()`. An algorithmic clear uses a declared Input
and an ordinary State transition; see [S9](../model-semantics.md#s9-reset).

The same inputs, Schedule and control settings after
reset produce the same results. Custom state, including random-generator
state if applicable, remains the author's reset responsibility.

<a id="method-wrapping-and-inheritance"></a>

### Method wrapping and inheritance

The author hooks are named `build`, `setup`, `reset`, `process`, `output` and `update`.
`__init_subclass__` collects declarations and wraps the setup/reset hooks.
`super().setup(ctx)` and `super().reset()` run base author bodies only;
framework steps occur once per outer lifecycle call. The guards must be
separate for setup and reset and must unwind on exceptions.
Setup invokes its final reset body under the reset-body guard too, so a
`super().reset()` there cannot trigger another recursive framework reset.

The generated constructor invokes the most-derived build body. A derived
build body can call `super().build()`. Processing hooks are ordinary author methods; callers use the framework
[entry points](execution.md), which check readiness and sequence.

<a id="whole-tree-finalization"></a>

## Whole-tree finalization

Locally configured children remain part of an unready tree.
An ancestor may still connect a child's boundary input during that ancestor's
setup. Root finalization resolves input forwarding to canonical external slots
or direct producer exports only after all local setup and ancestor wiring finish.
It must replace any provisional disconnected-input classification. Unbound
canonical external slots remain legal at setup and required before a consuming
Phase. A consumer checks the slots feeding that Phase's dependencies; it does not
require unused siblings' external inputs to be bound merely to run that consumer.

Building the final input bindings must not depend on sibling registration order.
A signal cycle with distinct writers is valid: endpoint resolution stops at
producer output storage and does not recursively follow that producer's inputs.
Output forwarding itself travels upward through the hierarchy and has no signal
cycle. Late ancestor wiring can change a child's input source, not whether that
child's own output was allocated or forwarded.

Author setup/reset hooks may initialize their own State and inspect
locally initialized child properties. They must not rely on connected input values
before root finalization. Public run/reset on the pending tree remain forbidden.
Setup failure has no rollback and leaves the tree unusable. Finalization is not
a public live-rewiring operation or a setup-time execution pass.

After successful setup, rebinding a canonical external slot validates the new
array against the complete root writer list before changing any readers. Either
all forwarded readers observe the new binding, or rejection preserves every old
binding/view. Reusing the original caller array remains a no-op. These are
observable requirements, not a required cache representation.
