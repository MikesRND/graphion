# Declarations

[Programming reference](index.md)

<a id="field-declarations"></a>

## Declarations

| Declaration | Meaning |
|---|---|
| `p: T = Param(default=value)` | Fixed configuration; omit `default` for a required argument. |
| `k: T = Control(default=value)` | Mutable configuration between Block Iterations. |
| `s: T = State(default=value)` | Scalar State; no axes and no dtype argument. |
| `s: Array = State(*axes, default=value, dtype=Sample)` | Array State; one or more axes. |
| `x: Array = In(*axes, dtype=Sample)` | Borrowed input array, read through its attribute. |
| `y: Array = Out(*axes, dtype=Sample)` | Fixed writable direct output or borrowed forwarded output. |

Axes are positional. `default` and `dtype` are keyword-only. Ports accept no
default. State requires a default. `Param` and `Control` accept no axes or dtype.
Port and array State axes support [exact dimension expressions](context-and-types.md#exact-dimension-expressions)
such as `N / factor`, where `factor` directly references a fixed Param declaration.
Zero-axis ports carry zero-dimensional arrays. Zero-axis State is scalar;
zero-dimensional array State is not part of this Implementation Profile.

<a id="construction-and-configuration"></a>

### Construction and configuration

- The generated constructor accepts Params and Controls as keyword-only
  arguments. A declaration without a default is required.
- State and ports are not constructor arguments.
- After assigning configuration, the constructor calls optional `build()`.
  It runs once and can read Params and Controls, but there is no ctx yet.
- Normal blocks do not define `__init__`. An author-provided `__init__`
  is rejected; construction and framework initialization are provided
  by Block.
- Scalar configuration is stored as supplied; the declaration describes
  its intended type. There is no general coercion or validation language.
  Authors validate domain constraints in `build()` or `setup()`.
- Params and Controls whose resolved annotation is `np.ndarray`, or whose
  origin is `np.ndarray` after following NumPy's `NDArray` alias, are independent
  host NumPy arrays. This includes `Array` and `NDArray[np.float64]`; unions and containers do
  not qualify. Defaults are copied per instance, and assigning an
  array Control makes an independent copy. Array Params and Controls are made read-only
  to prevent accidental mutation. Change array Controls by assignment. Array Params belong to host configuration;
  device coefficient storage, when supported, is prepared separately by
  the block during setup.
- Array configuration must convert to a NumPy array without Python-object
  storage (`dtype.hasobject` must be false). This rejects `None`, object arrays,
  structured arrays containing object fields, and values such as ragged lists
  that fail conversion. A bad declared default raises `DeclarationError` at
  class definition, naming the field. A bad constructor value or Control
  assignment raises `ParamError`; a rejected assignment preserves the previous
  Control array and its contents. Mutable defaults are checked again when used
  to construct an instance. Rectangular array-like values and numeric scalars
  remain accepted; the latter produce zero-dimensional configuration arrays.
- Other mutable configuration objects have author-defined ownership. The
  framework does not recursively copy arbitrary Python object graphs.
- Controls are not ports. A composite forwards exposed scalar Controls
  by assigning them to its children at the start of `process()`. This
  pattern covers scalars only: array-valued control sharing or copying is
  the author's responsibility. Signal-valued modulation is a declared In.
  There is no control-change hook or automatic propagation. Split output hooks
  must not assign Controls; establish Child Controls before opening the iteration.
  Assignment to a Block with a pending iteration, or any of its descendants, raises
  `ParamError`; see [execution](execution.md#controls-and-failures).
- Structure and interfaces depend on Params and ctx. A Control must not
  determine children, connections or port and State dimensions, so
  changing a Control never alters a configured interface. The framework
  does not detect a violation.

For example, `cutoff: float = Control(default=1000.0)` remains visible to tools and
can change at runtime. It does not become an input edge in the diagram.

<a id="state"></a>

### State

- Scalar State with no axes holds a scalar. Ordinary assignment updates it.
- Array State is allocated once its dtype and dimensions are resolved.
  A scalar default is converted to the declared dtype and fills the
  array; an array default must match its resolved shape and dtype and is
  copied without broadcasting. Unsupported scalar conversions raise
  `DtypeError`; a complex scalar default requires a complex State dtype.
  Instances never share the default's storage.
- State declarations require a default. Other Python objects can be
  ordinary attributes managed by the author's reset hook.
- Array State is updated in place and keeps its storage across runs and
  reset. Replacing its buffer after setup raises `StorageError`. Scalar
  State remains assignable. State is private storage, not a connection.
- Reset restores declared defaults. An optional author `reset()` body can
  then set initial conditions derived from parameters or ctx.
- A fixed array default is valid when its shape matches the configured
  instance. For context-dependent, nonuniform initial conditions, use a
  scalar declared default and initialize values in the reset body; there
  are no default factories.
- Initialized State is available in `reset()` and every processing hook. Setup
  computes sizes and coefficients; reset establishes initial state values.
- An output must not share storage with mutable State. Copy state values
  into the output buffer when returning them; this needs no new array.
  Likewise, copy retained input samples into preallocated history.
- Reusable scratch arrays that do not need reset can be ordinary
  attributes allocated in `setup()`. Scratch is workspace whose previous
  contents do not affect subsequent behavior. It needs no additional
  declaration kind. Avoid assigning a temporary to a declared State attribute.

<a id="declaration-syntax-and-inheritance"></a>

### Declaration syntax and inheritance

Fields hold declaration roles, axes, dtype and defaults. Annotations describe
instance values. The public `Array` alias is `numpy.typing.NDArray[np.generic]`;
it does not encode shape, actual sample dtype, writability or Param immutability.
Those remain runtime contracts. `dtype` defaults to `Sample` for ports and array
State, irrespective of the value annotation.

The typing surface must support keyword-only generated Params/Controls and
exclude State and ports from construction, using standard field specifier
metadata compatible with `dataclass_transform`. A block is not thereby a runtime
dataclass. The `init` parameter on `In`, `Out` and `State` is reserved for
typing metadata that excludes them from the generated constructor. Passing
`init` explicitly, including `init=False`, raises `DeclarationError` at runtime.
There is no author-selectable constructor participation, equality or dataclass
serialization API.

Declarations are collected through single inheritance. An override must be an
annotated field of the same kind. Replacing an inherited declaration with a
plain value, method or property, or redeclaring only its annotation, raises
`DeclarationError` at class definition.

A redeclaration replaces the inherited declaration completely, including its
default, while keeping its position in declaration order. For example,
`gain: float = Control()` is required even if the parent gives `gain` a default.
An untouched inherited declaration retains its default and order.

Names may not collide with the framework API. Child-name conflicts are checked
at [registration](lifecycle.md#children). Names beginning with `_` are
reserved for implementation details. The author hooks are `build`, `setup`,
`reset`, `process`, `output` and `update`; framework-owned `__init__`,
`run`, `run_output`, `run_update`, `bind_output`,
`schema`, `ctx`, `child`, `connect`, `children`, `connections` and `describe` cannot be overridden. The implementation lists reserved
names explicitly. Required constructor arguments remain keyword-only, so
an inherited default does not affect their order.

Multiple inheritance of Block subclasses is unsupported.

<a id="annotation-resolution"></a>

## Annotation resolution

The framework obtains each field's value type from its resolved annotation,
not from its default or field object. Already-resolved type objects are accepted
directly. Python 3.14 deferred annotations preserve the defining scope available
through Python's annotation machinery; they are not converted to source strings
and reinterpreted in module scope.

Remaining string or forward-reference annotations resolve with the semantics of
`typing.get_type_hints`, using the defining module globals and the declaring class
namespace (including its own name, inherited names and PEP 695 type parameters).
Within that supported scope, evaluated, quoted, postponed and deferred forms
produce equivalent declaration metadata. Inherited declarations retain their
resolved types from their declaring scope. Implementations may use an equivalent
resolver; they must not require unrelated ordinary-attribute annotations to resolve.

A missing field annotation or unresolved referenced type raises `DeclarationError`
at class definition, naming the class and field. Unquoted function-local types
that Python can resolve are supported. The framework does not reconstruct local
namespaces for quoted or postponed annotations whose names are absent from the
supported resolution scope. Forward references to not-yet-defined classes are
unsupported; use types available when the block class is defined. `describe()` uses the stored
resolved value type and never reevaluates annotations or axis properties.

Ports and array State must be annotated `Array` (an alias resolving to the same
`NDArray[np.generic]` type is equivalent). A scalar annotation on array State, or
`Array` on zero-axis scalar State, raises `DeclarationError` at class definition.
Scalar State annotations must designate a Python numeric/bool scalar or NumPy
numeric/bool scalar type; arbitrary object history belongs in author attributes.
Using `dtype` for scalar State or any unsupported field argument is a declaration
error. Scalar values are stored as supplied; these rules do not introduce scalar
coercion or runtime value-type checking.
