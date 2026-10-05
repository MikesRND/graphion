# Introspection

[Programming reference](index.md)

<a id="introspection-and-structural-description"></a>

## Introspection and structural description

The framework exposes:

- A read-only class `schema` mapping from declaration names to the records below,
  in inherited declaration order; overrides retain their position.
- A read-only `children` mapping of registered child names to child objects.
- An immutable `connections` view in recording order, containing this owner's
  [connection records](ports-and-connections.md#recording-a-connection).
- `describe()`, which returns a structural description of the tree.

### Class schema records

Each `schema` value exposes these public attributes:

| Attribute | Value |
|---|---|
| `name` | Declaration name string. |
| `kind` | One of the strings `Param`, `Control`, `State`, `In`, `Out`. |
| `value_type` | The resolved annotation object. |
| `dtype` | `None` for Params, Controls and scalar State; otherwise the declared dtype selector, including the default `Sample`, without context resolution or normalization. |
| `axes` | `None` for Params, Controls and scalar State; otherwise a tuple of declared axes, empty for a zero-axis port. |
| `default` | The declared default object, or `graphion.MISSING` for ports and required configuration. |

`MISSING` is a public singleton, tested by identity, distinct from a default of
`None`; its representation is `MISSING`. Schema records compare by identity,
returning a boolean without comparing defaults. To compare declaration metadata,
inspect the attributes explicitly. Record attributes cannot be reassigned. The default object itself is
neither copied nor frozen by the record and may be mutable. Instance storage
still follows the per-instance copying rules for array configuration and State.
Schema records describe declarations; they do not change when an instance is
configured, set up or reset. They are separate from the plain-data records
returned by `describe()`.

### Resolved port specs

`block.port_spec(name)` returns an immutable `PortSpec` with `shape` (a tuple of
integer dimensions) and `dtype` (a NumPy dtype). `PortSpec` is exported from
`graphion` and `graphion.profile.numpy_frame`. The record is cached; reading it
does not allocate signal storage or access the port's array.

Specs are available after successful local setup, including for unbound inputs
and forwarded outputs. A parent may read an explicitly configured child's spec
inside its own setup body, before whole-tree finalization:

```python
self.stage.setup(ctx)
output_length = self.stage.port_spec("y").shape[1]
```

An unknown or non-port name raises `SetupError`. Reading a spec before
local setup succeeds raises `SetupError`. A subsequent whole-tree failure does
not erase locally resolved specs. Specs contain no sample rate, timing contract
or execution settings. `port_spec` is a reserved framework-owned member.

The initial author `reset()` hook runs before local setup completes. It can
access allocated arrays but cannot yet call this block's `port_spec()`. Later
explicit reset calls can read the configured specs.

<a id="the-structural-description"></a>

### The structural description

`describe()` returns plain data: `None`, booleans, numbers, strings,
lists and mappings, with no block objects, arrays, modules or callables.
It can be consumed without the runtime that produced it.

The exact keys and representations are defined once in [the description schema](#description-schema).
The export includes hierarchy, definition identities, Params/Controls, all
declarations, endpoint records and resolved Implementation Profile facts.

This entire serialization is specific to the Python/NumPy Implementation Profile.
Shared concepts do not prescribe these keys or ndarray representation facts.
State is described by its declaration only, not its update rule or contents.
Every block in this Implementation Profile reports its execution ownership as
`opaque` ([shared block model](model.md#shared-concepts)).

**Completeness.** Topology is one fact for the whole described subtree.
It is `open` until the lifecycle root's setup has succeeded: the
children and connections recorded so far are reported. Open means completeness
is not certified; it does not promise further mutation after local or failed setup. Successful root setup closes
registration and wiring throughout the tree and makes the described subtree
`complete`. Failed setup remains `open`; locally configured children in an
unready tree are not reported as complete.
Interface resolution is separate. It is reported per dtype and per
dimension, each as a resolved value or in its declared symbolic form:
`Sample`, `C`, `N`, a named `Axis`, or a dimension expression. `declared_shape`
retains the declaration at every lifecycle stage; `shape` reports resolved
dimensions after local setup. Describing a block does not evaluate expressions
or axis properties.

**Identity.** A path such as `root/a` identifies a block within one
described structure; renaming or moving a block changes it. Definition
identity is the class's module and qualified name. It names the
definition and is not promised to be an importable reference.
Public re-exports retain the defining module: the library's `Gain` reports
`graphion.dsp.blocks.arithmetic`. Type records may name private
Implementation Profile modules, such as
`graphion.profile.numpy_frame._declarations` for `Axis`.

**Values.** Configuration values distinguish literals, array summaries and
opaque objects under the value records in [the description schema](#description-schema).

The description is an inspection snapshot, not a checkpoint or a
reconstructible model; signal arrays and State contents are excluded. A
tool that attaches its own data keys it by path and definition identity;
the core has no annotation registry.

<a id="diagram-scope"></a>

### Diagram scope

An exporter can draw either a collapsed block or its children and boundary
ports. It does not have to inspect Python source, process a frame or match
array identities.

Diagram scope is explicit:

- Recorded connections are the edges used for runtime input lookup and
  boundary output forwarding.
- Arithmetic, private state updates and ordinary helper calls inside a
  leaf are not automatically expanded. Feedback therefore appears as a
  stateful node, not an inferred internal feedback diagram.
- A diagram shows topology, not an inferred schedule, call count, time
  delay or proof that the user's processing order is correct.
- Signal-flow cycles are neither searched for nor rejected. An attempted
  read of an unset external input can still fail. Reading an allocated
  output before it contains meaningful data is a scheduling error the
  framework does not detect.

Diagram renderers consume the structural description. Rendering and
file formats are outside the Graphion runtime.

<a id="description-schema"></a>

## Description schema

`describe()` returns the following record vocabulary, made of JSON-compatible
plain values. Required keys below
are always present; optionality is expressed by `null`, not missing keys.
No extra keys are emitted in schema version 1. This is an inspection snapshot,
not a reconstructible input format. Unknown schema versions require a consumer
compatibility decision; they must not silently be treated as version 1.

### Root record

| Key | Value |
|---|---|
| `schema_version` | Integer `1`; identifies the serialized format, independently of the package version. |
| `profile` | Implementation Profile identifier: string `python.numpy.frame`. |
| `topology` | `open` or `complete` as defined in [the structural description](#the-structural-description). |
| `blocks` | Block records, preorder with children in registration order. |
| `connections` | Connection records, owners in block order and edges in recording order. |
| `unresolved` | Unresolved records, ordered by block then declaration order. |

Paths are relative to the described block, named `root`. Child components are
registration names with `~` escaped as `~0` and `/` as `~1`, joined with `/`.
The path and definition identify a snapshot node; neither is a persistent UUID.
The description includes only connection records owned inside this subtree.
A subtree boundary omits incoming wiring owned by an outside ancestor; topology
completeness refers only to the exported subtree, not its surrounding model.

### Block record

| Key | Value |
|---|---|
| `path` | Hierarchy path string. |
| `definition` | Object with `module` and `qualname` strings for the class. |
| `execution_ownership` | `opaque`. |
| `phases` | Mapping from processing hook names to their required Input names. |
| `parameters` | Mapping from Param names to represented current values. |
| `controls` | Mapping from Control names to represented current values. |
| `declarations` | All declaration records in inherited declaration order; overrides keep their slot. |
| `children` | Direct child path strings in registration order. |
| `context` | `null` until local setup succeeds; then an object with `rate`, `rate_exact`, `n`, `channels`, `dtype` and `xp`. |
| `storage` | `null` until root setup succeeds; then a mapping from each port/State name to a storage-role string. |

For example, FrameDelay reports `{"output": [], "update": ["x"]}`. Atomic
`process` and State `update` list every declared Input. `output` lists all Inputs
unless narrowed by `@needs_inputs(...)`; names follow schema declaration order.
A declaration-only Block reports `{}`. These are declared dependencies, not proof of hook behavior or
Schedule correctness; see [S12](../model-semantics.md#s12-descriptions-and-claims).

Context `dtype` is NumPy `dtype.str`; `xp` is `numpy`; `n` and `channels` are
plain JSON integers. The rate fields are always present:

- `rate` is `float(ctx.rate)`, a positive finite JSON number whose range was
  validated when `Ctx` was constructed. It may be an approximation.
- `rate_exact` is `{"numerator": integer, "denominator": integer}` for an integer
  or Fraction rate. Integer rates use denominator `1`; Fraction records use the
  reduced fraction with a positive denominator. Floating rates use `null`, even
  when their numeric value is an integer; no exact fraction is inferred from a
  float. NumPy scalars follow their Python type after context conversion.

Consumers requiring exact rational rates must preserve the precision of these
JSON integers. Storage roles are `direct_output`, `supplied_output`,
`forwarded_output`, `connected_input`, `external_input`, `array_state`, or
`scalar_state`. A boundary input connected outside the exported subtree retains
`connected_input`; no outside node or edge is fabricated. Configuration arrays
are represented as values, not writer-list entries.

### Declaration record

| Key | Value |
|---|---|
| `name` | Declaration name string. |
| `kind` | `Param`, `Control`, `State`, `In` or `Out`. |
| `value_type` | Type record defined below, from the stored [resolved annotation](declarations.md#annotation-resolution). |
| `dtype` | `null` for configuration/scalar State; otherwise a dtype record. |
| `shape` | `null` for configuration/scalar State; otherwise a list of dimension records (empty for a zero-dimensional port). |
| `declared_shape` | `null` for configuration/scalar State; otherwise the original dimension records, retained after setup. |
| `default` | Represented declared default; `{"kind":"required"}` for required configuration; `null` for ports. |

A type record is `{"kind":"array"}` for `Array`; a named Python type is
`{"kind":"named","module":string,"qualname":string}`. Other typing forms use
`{"kind":"opaque","type":string}` identifying the runtime type of the resolved
annotation object, by module plus qualified name. That intentionally does not
claim a portable generic type language. `np.ndarray` configuration annotations
are named types. No `repr()` of arbitrary objects is used.

A dtype record is `{"kind":"resolved","value":string}` using `dtype.str`, or
`{"kind":"symbol","name":"Sample"}`. A dimension record is
`{"kind":"resolved","value":integer}`, `{"kind":"symbol","name":"C"}`,
`{"kind":"symbol","name":"N"}`, or `{"kind":"axis","name":string}`.
Dimension expressions use the recursive record
`{"kind":"expression","operator":"multiply"|"divide","left":operand,"right":operand}`.
An operand is another expression, a `C`/`N` symbol, a validated Param reference
`{"kind":"param","name":string}`, or an exact rational constant
`{"kind":"fraction","numerator":integer,"denominator":positive_integer}`.
Rational constants are reduced; integer expression constants have denominator
one. Param references identify fields on the described instance, including
inherited fields and overrides; no Python object identities are exported.

Before successful local setup, literal valid dtypes/axes are represented directly;
context tokens and named axes stay symbolic even if a ctx has temporarily been
assigned. Export must not evaluate axis properties, allocate buffers or invoke
setup/reset/process. An invalid literal selector is represented as
`{"kind":"invalid","type":string}` and listed in `unresolved`; export does not
perform setup validation. After successful local setup, array specs are resolved.
A later failure of the overall tree does not erase already resolved local facts.
`declared_shape` always preserves the original symbols, named axes, literal
dimensions and expression trees; `shape` then uses cached resolved integers.
Symbolic entries in `declared_shape` alone do not create unresolved facts.

Literal dtype inspection accepts NumPy dtype objects, built-in scalar types and
values, NumPy's concrete scalar types and values, and plain list/tuple/dict
selector forms composed of those literals. NumPy normalizes these forms to a
numeric dtype, or they are reported as invalid. Inspection does not invoke
arbitrary selector attributes or representations. An opaque or cyclic selector
is reported as invalid before local setup; after successful setup the stored
resolved dtype is reported without reevaluating its selector.

### Values

| Value | Record |
|---|---|
| `None`, bool, int, finite float, str | `{"kind":"literal","value":value}` |
| Python `Fraction` | `{"kind":"fraction","numerator":integer,"denominator":positive_integer}` |
| Complex | `{"kind":"literal","value":{"real":number_or_tag,"imag":number_or_tag}}` |
| Nonfinite float | `{"kind":"literal","value":{"float":"nan"}}`, with tag `nan`, `+inf` or `-inf`. |
| Supported NumPy scalar | Corresponding literal record plus `dtype: dtype.str`. |
| ndarray | `{"kind":"summary","dtype":dtype.str,"shape":[integers]}` |
| Other value | `{"kind":"opaque","type":module_plus_qualified_type_name}` |

`number_or_tag` is a finite float or the same single-key `float` tag object.
NumPy scalars are literal only if their `.item()` value is one of the supported
Python scalar types above; otherwise they are opaque. Required/default tags are
schema records, not user-value dictionaries. Lists, mappings, callables and
arbitrary configuration objects are opaque, not recursively traversed. Neither
signal arrays nor current State values are included. Array State defaults may be
summarized; scalar State defaults may be literal. Summaries do not retain arrays.

Integer subclasses, including `IntEnum`, use integer literals without member
names. An enum class used as the field annotation is still identified by its
declaration's `value_type` record.

### Connections and unresolved facts

A connection is
`{"owner":path,"source":{"path":path,"port":name},"destination":{"path":path,"port":name}}`.
It is taken from the same endpoint record used for runtime lookup, never inferred
from shared storage. All referenced blocks and ports must occur in the export.

An unresolved record is
`{"path":path,"declaration":name_or_null,"fact":string,"reason":string}`.
`fact` is one of `topology`, `context`, `storage`, `dtype`, `shape`, or `value_type`.
Report open topology once at `root`; report missing context/storage per block,
symbolic or invalid dtype/shape per affected declaration, and opaque value types
per declaration. Reasons are explanatory text, not executable semantics. An
empty State declaration list is not an assertion of purity or full State exposure.

The [configured Gain example](description-example.json) fixes a concrete
serialization example.
