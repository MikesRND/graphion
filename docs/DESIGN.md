# Graphion design

[Model Semantics](model-semantics.md) defines shared behavior; the
[programming reference](reference/index.md) defines the NumPy API. This document explains the architecture and selected tradeoffs;
the [charter](CHARTER.md) defines the enduring principles. Implementation status
is reported separately in the [README](../README.md).

## General model, bounded implementation

Graphion serves general processing models; RF/DSP is one application. The shared
concepts describe interfaces, configuration, state, ownership, invocation and
completion without requiring arrays or sample clocks. Graphion implements a
synchronous Python/NumPy frame Implementation Profile.

## Implementation Profiles and application libraries

The
`graphion.profile.numpy_frame` package implements the Python/NumPy frame
Implementation Profile, identified as `python.numpy.frame` in structural
descriptions. It provides synchronous, caller-ordered execution and NumPy storage.
The root `graphion`
package re-exports its authoring API.

Application libraries have their own public names. `graphion.dsp.blocks` provides
DSP blocks grouped by family; its implementations support the Python/NumPy frame
Implementation Profile. Library names do not imply support for every
Implementation Profile. Additional host languages, backends, or models of
computation require explicit Implementation Profile contracts.
There is no plugin registry or automatic implementation selection.

`Ctx` and the `C`, `N`, and `Sample` declaration tokens belong to the Python/NumPy
frame Implementation Profile. Context configuration is specific to that
Implementation Profile; it is not a universal description of a block's input
and output signals.

Implementation Profile modules and DSP block modules do not import the root
convenience API. Storage operations take resolved shapes and dtypes and depend on neither
declaration tokens nor `Block`. Calls to private helpers go through their module
attributes so that verification guards intercept the functions used during
execution.

## Field declarations

Value annotations describe what an author reads. Field specifiers identify role,
dimensions, dtype and defaults: `x: Array = In(C, N)` and
`gain: float = Control(default=1.0)`. This choice gives
useful static constructor and attribute checking. Axes are positional; dtype is
keyword-only and defaults to `Sample` for arrays. Runtime validation retains the
precise dimensions and storage rules that broad static array types cannot express.

## Explicit execution and visible structure

Children and connection records expose structure without granting scheduling
ownership. A composite's process body may forward controls, run a child several
times or skip it. Flattening that body into a graph traversal can change behavior.
Consequently, ordinary blocks remain opaque execution units to external tools.
The build boundary serves the charter's visible-structure principle by fixing an
inspectable diagram independently of execution configuration.

Connections are the shared authority for runtime lookup and diagrams. Memory
aliases and observed execution traces cannot reliably describe the intended
model. The [execution contract](reference/execution.md) keeps scheduling in
ordinary author/caller code. Feedback behavior follows those calls and explicitly
initialized State. Conservative defaults and explicit Output dependency decorators describe readiness without
authorizing tools to reorder opaque processing bodies. Split output/update calls
let State break a feedback cycle without making Ports carry history.

## Predictable storage

The NumPy Implementation Profile uses fixed output storage, private history,
borrowed input views and cached result views. This permits allocation-conscious
kernels without requiring different interfaces from convenient allocating expressions. Hierarchy
forwarding adds no signal buffer. Retention remains an explicit copy or private
history update.

Read-only views catch accidental writes; they are not security barriers or
snapshots. Binding validates shape, dtype and storage independence. A conservative
stride condition makes supplied writable layouts checkable without promising
acceptance of every mathematically non-overlapping arrangement. These precise
rules live in [storage and lifetime](reference/storage-and-lifetime.md).

## Construction, build and whole-tree finalization

Construction assembles a block's initial structure. Build commits the assembled
model's topology. Setup prepares that model for execution. The constructor runs
the author build body so a composite is a fully constructed object that parents
and callers can compose immediately. Deferring assembly would add partially
constructed objects without strengthening the freeze guarantee.

Build fixes the topology before any setup. Explicit child setup allows derived
contexts and child-dependent dimensions. Local setup initializes child storage
and State. The root resolves final input bindings and validates the writer set
before making any descendant runnable. This supports sibling feedback without
confusing endpoint resolution with signal scheduling.

Build rejection is nonfatal because it runs no author hooks and allocates no
signal or State buffers; all structural checks precede the freeze. Failed setup
leaves the tree unusable. There is no transactional rollback of author hooks.
The [lifecycle reference](reference/lifecycle.md) specifies the ordering and the
atomicity that is required for later external-input rebinding.

## Inspection and conformance

The Python description is a versioned inspection snapshot, not a checkpoint or
loadable model. Known interface facts, topology completeness and opaque behavior
are separate facts. Other Implementation Profiles need not use this serialization.

The reference defines conformance. The tests check the behavior they cover,
not the complete specification. Measured performance is workload-specific,
not a portable timing gate.

## Outside this Implementation Profile

The Python/NumPy frame Implementation Profile does not include output replacement,
per-call output destinations, partial frames, live rewiring, processing that
overwrites its own input, automatic scheduling, parallel execution, checkpointing, diagram
rendering, a public backend framework or a model compiler. Adding any of these
requires a concrete use and an explicit contract for its storage and execution
consequences, and must keep ordinary block authoring simple.

## Future directions

The project expects to grow in these directions. None is designed or scheduled
yet; each is listed because it constrains choices made today.

- **Checkable composition.** The charter's [Purpose](CHARTER.md#purpose)
  describes execution automated under a model of computation. Connection
  records stay the wiring authority, and a description keeps separating declared
  structure from opaque behavior.
- **Other Implementation Profiles.** Purpose names C++, CUDA and HDL. The shared
  model therefore assumes neither synchronous completion nor NumPy storage; a
  GPU or asynchronous Implementation Profile states when work and transfers
  complete.
- **Numeric formats.** A logical format such as fixed-point width and scale is
  distinct from storage dtype, with its own rounding and overflow rules.
- **Partial frames.** Valid lengths within fixed-capacity storage, not
  reallocation per call.
