# Storage and lifetime

[Programming reference](index.md)

<a id="external-inputs-and-output-storage"></a>

## External inputs and output storage

- An unconnected In retains the caller-supplied array and exposes a
  cached read-only view through its Input attribute. Assigning `block.x = frame` validates and
  replaces the input binding and its view. The caller can still refill
  its own original array between calls.
- A connected In reads its recorded source's read-only export. A fixed
  output source resolves to a cached view after wiring is configured.
  Input forwarding reads the current external input slot and observes
  replacements. Assigning to a connected In raises `WiringError`.
- A directly produced Out has a writable ndarray after setup, with fixed
  identity, shape and dtype. Before allocation, reading it raises
  `UnavailablePortError`.
- The author writes into that storage with `out=self.y`,
  `np.copyto(self.y, value)`, or `self.y[...] = value`.
- Assigning a different array to an Out raises `StorageError`;
  `self.y = expression` does not implicitly copy into it. Use an in-place
  write. An external caller may provide storage with `bind_output()`
  before allocation as described below.
- Assigning the identical array object back to an Out or array State is
  a no-op. This permits NumPy's ordinary augmented operations such as
  `self.y *= gain`, which mutate the array and then assign that same object
  back to the attribute. It does not permit changing the buffer identity.
- A forwarded Out exposes the direct producer's cached read-only export,
  even through several hierarchy boundaries. Only the direct producer's
  own Out attribute exposes writable storage.

An allocated output buffer is not necessarily initialized signal data.
Its contents are unspecified until the producer has filled it. Consumers
must be scheduled after the work that produces the values they need. The
framework does not inspect array contents or prove that every element was
written.

Here, "owned output" means that the direct producer controls writes and
result lifetime. Its storage can be framework-allocated or supplied by a
caller; this term does not require NumPy's `OWNDATA` flag to be true.

The external input slot is a borrowed current value, not retained history.
The caller can reuse its buffer between calls. An algorithm that needs old
samples stores its own copy in State.

Read Input attributes through their current bindings in each processing hook.
Do not retain an input reference as algorithm history or assume that it
follows a later rebinding. Local references within one invocation are
fine. This keeps input access behind the binding, so a different
execution environment could supply input storage without changing block
code.

<a id="supplying-output-storage"></a>

## Supplying output storage

By default, the framework allocates each directly produced Out once during
setup. An owner or host can supply a buffer instead:

```python
amp = Gain(gain=0.5)
amp.bind_output("y", destination)
amp.setup(ctx)
```

`bind_output(name, array)` is allowed before setup, including after build, or in
that block's author setup body. It supplies storage, not initial signal values. The
framework checks exact shape, dtype, writability and the [supplied-output layout condition](#supplied-output-layouts) after specs resolve,
then uses the supplied buffer without allocating another output array.

Bind the direct producer's Out. A forwarded boundary Out has no separate
storage to bind; its owner can bind the producing child's Out before
setting that child up. Output storage cannot be replaced after setup.

Binding storage to a forwarded Out raises `StorageError` at `bind_output()`
if its forwarding connection already exists. Recording a forwarding connection
to an Out with supplied storage raises `StorageError` at `connect()` instead.
Either rejection preserves the existing buffer binding and connection records.

Framework-allocated output and State buffers are fresh and distinct.
Supplied outputs must be disjoint from every other declared writable
buffer in the lifecycle tree. External inputs must be disjoint from all
those writable buffers. [Tree-wide overlap validation](#tree-wide-overlap-validation) specifies the checks. Disjoint slices
of a larger storage bank are allowed. Processing that overwrites its own
input requires an aliasing contract outside this Implementation Profile.

<a id="tree-wide-overlap-validation"></a>

## Tree-wide overlap validation

Before the outermost setup succeeds, collect one list of declared writable
storage for the entire registered lifecycle tree: every directly produced
Out, whether allocated or supplied, and every array State. Record the
owning block path, declaration name, buffer and whether it was supplied.
Connected inputs and forwarded exports are readers, not additional writers,
so they do not create entries. Do not deduplicate distinct writer slots by
array identity: binding one array to two producers must be caught.

At this finalization point:

- Check each supplied output against every other writer entry, excluding
  only its own declaration slot. Pairs of freshly framework-allocated
  buffers need no overlap check against each other.
- Check every external input already supplied, including those assigned
  before setup (including after build) or inside setup bodies, against the complete writer list.
  Inputs without a value remain allowed until processing.

Structural self-aliases are rejected earlier by [build](lifecycle.md#build) with
`WiringError`. Actual buffer overlap remains a `StorageError` during storage validation.

After successful setup, assigning a different array object to any external
input checks it against the same whole-tree writer list, before changing
the binding. A failed assignment preserves the previous input and view.
Refilling the original input needs no assignment; reassigning the identical
retained caller array is a no-op. Neither case repeats checks or makes a
new view. Rebinding a child input uses its lifecycle root's list too.

Host feedback such as `fx.x = fx.y` is rejected because the external input
would alias storage that the same tree writes. Use algorithm State or a
separate preallocated host history buffer into which the host copies the
result before the next call.

For each required memory comparison, first use `np.may_share_memory`.
On a possible overlap, use `np.shares_memory` with a finite positive work
limit. Reject confirmed overlap or exhaustion of the work limit with
`StorageError`; distinguish those reasons and name both storage roles in
the error. Accept disjoint strided slices that the exact check resolves.

This covers the tree's declared output/array-State storage and its external
input bindings, under the fixed-array-metadata contract. Ordinary author
scratch, arbitrary configuration objects and buffers belonging to other
lifecycle trees are outside the check. Sharing those remains the author's
responsibility. The writer list validates storage; endpoint records
define every connection. These checks do not add a per-run overlap scan.

<a id="the-buffer-contract"></a>

## The buffer contract

- Inputs, forwarded outputs and `run()` results expose borrowed read-only
  views. The direct producer writes through its own Out attribute; the
  framework never toggles that buffer's flags around execution.
- Producer buffer identity is fixed across runs and reset.
- Exported view identity is fixed across runs and reset.
- The exported view shares storage with the producer buffer.
- Input attributes and producer Output attributes need not be the same object.
- Array State also retains its identity across runs and reset. Each
  producer's outputs are disjoint from its own inputs and State.
- A Borrow expires immediately before the Direct Producer's next `process`
  or `output` hook is invoked, or when Reset begins within its scope.
  A readiness or sequence rejection before invocation preserves the Borrow;
  an invoked hook failure does not restore it. State Update preserves Outputs.
  Consumers must finish reads before expiration.
  `saved = block.y` is a reference, not a snapshot; use
  `saved = block.y.copy()` to retain a result.
- A forwarded output has the lifetime of its actual producer's buffer,
  not just the lifetime implied by calls to its containing composite.
- Fan-out shares the same producer buffer. The manual schedule must let
  every reader that needs the current value finish before the producer
  overwrites it. Queued or asynchronous consumers require a copy or an
  explicit buffer-ownership scheme; this synchronous core does not supply
  one implicitly.
- Copy values that must survive producer overwrites into private State.
  State updates remain part of the block's execution. Do not keep a
  reference to an input as a substitute for history. Use explicit State for
  [feedback across blocks](examples.md#feedback-across-blocks).
- Reset preserves storage and restores State. Output contents after
  reset are unspecified; Reset bodies must not write Outputs.
  The framework does not zero signal buffers or replace their arrays.

For a directly produced output, `block.run() is block.y` is false: the
first is the cached read-only view, the second is writable producer
storage. Repeated calls return the same view (or the same tuple of views).
Read-only views catch ordinary accidental writes. They are not snapshots
or protection against access through the base or writable producer
attribute. All writable aliases remain subject to the stability obligation in
[S8](../model-semantics.md#s8-output-stability-and-update-ordering). Storage visible
through any Port must not overlap State storage.

<a id="supplied-output-layouts"></a>

## Supplied output layouts

During setup, supplied output storage is accepted only if its elements pass
this conservative check for non-overlapping storage:

1. If any dimension has length zero, accept its layout (there are no elements).
2. Otherwise discard dimensions of length one and sort the remaining
   `(abs(byte_stride), length)` pairs by increasing stride.
3. Set `span = array.dtype.itemsize`.
4. For each pair, require `stride >= span`; if false, raise `StorageError`
   identifying the output and unsupported layout. Then set
   `span += (length - 1) * stride`.

All shape, dtype, writability and inter-buffer overlap checks still apply.
No per-run layout check is required under the fixed-metadata contract. This
additional rule applies only to supplied outputs. Framework-allocated outputs
and array State use fresh ordinary storage. Read-only external inputs may have
self-overlap; they must still pass input shape, dtype and writer-overlap checks.

This sufficient condition deliberately rejects some non-overlapping
interleavings. It accepts ordinary C/F-contiguous, transposed, reversed and
stepped layouts, including real/imaginary views of complex storage when their
other contracts hold. Use ordinary contiguous destination storage if rejected;
copy results afterward if a different destination layout is needed. This rule
does not validate whether an unsafe stride-trick view points inside a real
allocation; supplying a valid ndarray backing allocation is the caller's duty.

### Layout examples

The following examples use float64 elements, each eight bytes wide. Other
shape, dtype, writability and inter-buffer overlap checks still apply.

| Shape | Byte strides | Result |
|---|---|---|
| `(1, 4)` | `(0, 0)` | Rejected: all four elements share storage. |
| `(3, 2)` | `(8, 16)` | Rejected: two elements begin at offset 16. |
| `(2, 2)` | `(8, 4)` | Rejected: elements overlap by four bytes. |
| `(3, 2)` | `(16, 24)` | Rejected by the conservative rule, although elements are disjoint. |
| `(1, 0)` | `(0, 0)` | Accepted: an empty array has no overlapping elements. |

Writable reversed views such as `base[:, ::-1]`, transposes, ordinary stepped
slices and real/imaginary views pass the layout rule when their strides satisfy
it. A rejected supplied output raises `StorageError` during setup.
