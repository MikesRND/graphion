"""Block declarations, lifecycle and storage management."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from numbers import Integral
from types import MappingProxyType, MemberDescriptorType
from typing import dataclass_transform

import numpy as np

from graphion.errors import (
    DeclarationError,
    DtypeError,
    ExecutionError,
    ParamError,
    RegistrationError,
    SetupError,
    ShapeError,
    StorageError,
    UnavailablePortError,
    WiringError,
)

from . import _declarations, _dimensions
from . import _storage as arrays
from ._declarations import (
    MISSING,
    Axis,
    C,
    Control,
    In,
    N,
    Out,
    Param,
    Sample,
    State,
    _Declaration,
    _Field,
)
from .context import Ctx
from .spec import PortSpec

# Adding a public Block member requires an explicit collision-policy decision.
_FRAMEWORK_OWNED_NAMES = frozenset(
    {
        "__init__",
        "run",
        "run_output",
        "run_update",
        "bind_output",
        "schema",
        "ctx",
        "child",
        "connect",
        "children",
        "connections",
        "describe",
        "port_spec",
    }
)
_AUTHOR_HOOK_NAMES = frozenset({"build", "setup", "reset", "process", "output", "update"})
_RESERVED_DECLARATION_NAMES = _FRAMEWORK_OWNED_NAMES | _AUTHOR_HOOK_NAMES


@dataclass(slots=True)
class _TreeStatus:
    built: bool = False
    ready: bool = False
    failed: bool = False
    writers: tuple = ()


@dataclass(frozen=True, slots=True, eq=False)
class _Connection:
    source_block: object
    source_port: str
    destination_block: object
    destination_port: str


@dataclass(slots=True)
class _Call:
    hook: Callable[[], object]
    dependencies: tuple[str, ...]
    missing: tuple[str, ...] = ()


def _processing_phases(cls):
    hooks = {name: inspect.getattr_static(cls, name) for name in ("process", "output", "update") if hasattr(cls, name)}
    if "process" in hooks and len(hooks) != 1 or "update" in hooks and "output" not in hooks:
        raise DeclarationError(f"{cls.__qualname__}: define process, or output with optional update")
    for name, value in vars(cls).items():
        method = value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
        if inspect.isfunction(method) and hasattr(method, "_graphion_input_dependencies") and name != "output":
            raise DeclarationError(f"{cls.__qualname__}.{name}: needs_inputs applies only to output")
    inputs = tuple(port for port, field in cls.schema.items() if field.kind == "In")
    dependencies = {}
    for name, hook in hooks.items():
        if not inspect.isfunction(hook):
            raise DeclarationError(f"{cls.__qualname__}.{name}: processing hook must be an instance method")
        parameters = tuple(inspect.signature(hook).parameters.values())
        if (
            len(parameters) != 1
            or parameters[0].name != "self"
            or parameters[0].kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            or parameters[0].default is not inspect.Parameter.empty
        ):
            raise DeclarationError(f"{cls.__qualname__}.{name}: processing hook must take only positional self")
        declared = getattr(hook, "_graphion_input_dependencies", None)
        if declared is None:
            dependencies[name] = inputs
        else:
            for port in declared:
                if port not in inputs:
                    raise DeclarationError(f"{cls.__qualname__}.{name}: {port} is not a declared Input")
            dependencies[name] = tuple(port for port in inputs if port in declared)
    return MappingProxyType(dependencies)


def _wrap_build(body):
    @wraps(body)
    def build(self):
        if self._in_build_hook:
            return body(self)
        return self._build()

    return build


def _wrap_setup(body):
    @wraps(body)
    def setup(self, ctx):
        if self._in_setup_hook:
            return body(self, ctx)
        return self._setup(ctx, body)

    return setup


def _wrap_reset(body):
    @wraps(body)
    def reset(self):
        if self._in_reset_hook:
            return body(self)
        return self._reset(body)

    return reset


@dataclass_transform(
    field_specifiers=(Param, Control, State, In, Out),
    kw_only_default=True,
    eq_default=False,
)
class Block:
    """A declarative processing block with reusable storage and lifecycle children."""

    schema: MappingProxyType[str, _Field] = MappingProxyType({})
    _phase_dependencies: MappingProxyType[str, tuple[str, ...]] = MappingProxyType({})

    def __init_subclass__(cls):
        super().__init_subclass__()
        # Check mixins before declaration processing reads inherited schemas.
        for base in cls.__mro__[1:]:
            if base is object or issubclass(base, Block):
                continue
            overridden = _FRAMEWORK_OWNED_NAMES.intersection(base.__dict__)
            if overridden:
                raise DeclarationError(
                    f"{cls.__qualname__}: non-Block mixin {base.__qualname__} defines "
                    f"framework-owned names: {', '.join(sorted(overridden))}"
                )
        declarations = {name: value for name, value in cls.__dict__.items() if isinstance(value, _Declaration)}
        parameters = _declarations.dimension_bindings(cls, declarations)
        for name in declarations:
            if name in _RESERVED_DECLARATION_NAMES or name.startswith("_"):
                raise DeclarationError(f"{cls.__qualname__}.{name}: reserved declaration name")
        overridden = _FRAMEWORK_OWNED_NAMES.intersection(cls.__dict__)
        if overridden:
            raise DeclarationError(
                f"framework-owned names cannot be overridden: {', '.join(sorted(overridden))}; "
                "use build/setup/reset/process/output/update"
            )
        if sum(issubclass(base, Block) for base in cls.__bases__) > 1:
            raise DeclarationError("multiple Block inheritance is unsupported")
        fields = dict(cls.schema)
        annotations = _declarations.own_annotations(cls)
        for name in fields.keys() & (cls.__dict__.keys() | annotations.keys()):
            if name not in declarations:
                raise DeclarationError(
                    f"{cls.__qualname__}.{name}: inherited declaration requires an annotated same-kind field"
                )
        for name, declaration in declarations.items():
            previous = fields.get(name)
            if previous and previous.kind != declaration.kind:
                raise DeclarationError(f"{cls.__qualname__}.{name}: inherited declaration kind cannot change")
            value_type = _declarations.resolve_value_type(cls, name, annotations, declaration)
            if (
                declaration.kind in ("Param", "Control")
                and _declarations.is_array_configuration(value_type)
                and declaration.default is not MISSING
            ):
                try:
                    arrays.configuration_array(declaration.default, f"{cls.__qualname__}.{name} default", copy=False)
                except ParamError as exc:
                    raise DeclarationError(str(exc)) from exc
            field = _Field(
                name,
                declaration.kind,
                value_type,
                declaration.dtype,
                _declarations.bind_axes(declaration.axes, parameters, cls),
                declaration.default,
            )
            fields[name] = field
            setattr(cls, name, field)
        cls.schema = MappingProxyType(fields)
        cls._phase_dependencies = _processing_phases(cls)
        for name, wrap in (("build", _wrap_build), ("setup", _wrap_setup), ("reset", _wrap_reset)):
            # A hook owned by a Block ancestor is already a framework entry
            # point or wrapper; only this class's and mixins' bodies need wrapping.
            owner = next(base for base in cls.__mro__ if name in base.__dict__)
            if owner is cls or not issubclass(owner, Block):
                setattr(cls, name, wrap(owner.__dict__[name]))

    def __init__(self, **configuration):
        self._values = {}
        self._input_raw = {}
        self._input_readers = {}
        self._specs = {}
        self._port_specs = {}
        self._supplied = {}
        self._writers = ()
        self._exports = {}
        self._result = None
        self._phase = "new"
        self._local_setup_complete = False
        self._tree = _TreeStatus()
        self._parent = None
        self._child_name = None
        self._children = {}
        self._children_view = MappingProxyType(self._children)
        self._connections = ()
        self._incoming = {}
        self._ctx = None
        self._in_build_hook = False
        self._in_setup_hook = False
        self._in_reset_hook = False
        self._initializing_reset = False
        self._reset_prepared = False
        self._pending = False
        self._requires_reset = False
        self._calls = {
            name: _Call(getattr(self, name), dependencies) for name, dependencies in self._phase_dependencies.items()
        }
        # Share the phase readiness records refreshed by rebinding.
        self._process_call = self._calls.get("process")
        self._output_call = self._calls.get("output")
        self._update_call = self._calls.get("update")
        self._run_missing: tuple[str, ...] = ()
        self._config_locked = False
        self._missing_inputs = {name for name, f in self.schema.items() if f.kind == "In"}
        self._state_defaults = {}
        allowed = {name for name, f in self.schema.items() if f.kind in ("Param", "Control")}
        unknown = configuration.keys() - allowed
        if unknown:
            raise ParamError(f"unknown configuration: {', '.join(sorted(unknown))}")
        for name, field in self.schema.items():
            if name in allowed:
                value = configuration.get(name, field.default)
                if value is MISSING:
                    raise ParamError(f"{name}: required configuration was not supplied")
                self._assign(field, value)
            elif field.kind == "State":
                default = field.default
                if isinstance(default, np.ndarray):
                    default = arrays.host_copy(default, readonly=True)
                self._state_defaults[name] = default
        self._config_locked = True
        self._in_build_hook = True
        try:
            self.build()
        finally:
            self._in_build_hook = False

    def build(self):
        if not self._in_build_hook:
            return self._build()

    def _build(self):
        if self._parent is not None:
            raise SetupError("a registered child is built by its lifecycle root")
        if self._tree.built:
            return
        for node, name, source, port in self._input_sources():
            if source is node and source.schema[port].kind == "Out":
                raise WiringError(
                    f"{node._identity()}.{name}: connected input resolves to its own output writer "
                    f"{source._identity()}.{port}"
                )
        # Commit only after every structural endpoint has been validated.
        for node in self._walk():
            if node._phase == "new":
                node._phase = "built"
        self._tree.built = True

    @property
    def ctx(self):
        if self._ctx is None:
            error = SetupError(f"{self._identity()}: ctx is unavailable before setup")
            # Mark only framework context lookup errors for axis diagnostics.
            error._context_block = self
            raise error
        return self._ctx

    @ctx.setter
    def ctx(self, value):
        raise SetupError(f"{self._identity()}: ctx is managed by setup(); pass the context to setup()")

    @property
    def children(self):
        return self._children_view

    @property
    def connections(self):
        return self._connections

    def describe(self):
        """Return a plain-data description of this block and its descendants."""
        from . import _describe

        return _describe.describe(self)

    def port_spec(self, name: str) -> PortSpec:
        field = self.schema.get(name)
        if field is None or field.kind not in ("In", "Out"):
            raise SetupError(f"{self._identity()}.{name}: not a port")
        if not self._local_setup_complete:
            raise SetupError(f"{self._identity()}.{name}: port spec requires successful local setup")
        return self._port_specs[name]

    @property
    def _ready(self):
        return self._tree.ready

    @property
    def _unconfigured(self):
        return self._phase in ("new", "built")

    def _identity(self):
        parts = []
        block = self
        while block._parent is not None:
            parts.append(block._child_name.replace("~", "~0").replace("/", "~1"))
            block = block._parent
        path = "/".join(("root", *reversed(parts)))
        return f"{path} ({type(self).__qualname__})"

    def _walk(self):
        yield self
        for child in self._children.values():
            yield from child._walk()

    def _input_sources(self):
        for node in self._walk():
            for name, field in node.schema.items():
                if field.kind == "In":
                    yield (node, name, *node._endpoint_source(name))

    def child(self, name, block):
        if self._tree.failed:
            raise RegistrationError("cannot register children after setup failure; construct a new tree")
        if self._phase != "new":
            raise RegistrationError("register children before build")
        if not isinstance(name, str) or not name:
            raise RegistrationError("child name must be a nonempty string")
        if name in self._children or name in self.schema:
            raise RegistrationError(f"{name}: duplicate child name or declaration-name collision")
        if not isinstance(block, Block):
            raise RegistrationError(f"{name}: child must be a Block")
        if block._parent is not None:
            raise RegistrationError(f"{name}: child already has a lifecycle parent")
        ancestor = self
        while ancestor is not None:
            if ancestor is block:
                raise RegistrationError(f"{name}: lifecycle registration cycle")
            ancestor = ancestor._parent
        descendants = tuple(block._walk())
        if any(not node._unconfigured or node._tree.failed for node in descendants):
            raise RegistrationError(f"{name}: child must be previously unconfigured")
        # Validate before adopting the subtree, including its existing descendants.
        block._parent = self
        block._child_name = name
        for node in descendants:
            node._tree = self._tree
        self._children[name] = block
        return block

    def connect(self, source_block, source_port, destination_block, destination_port):
        if self._tree.failed:
            raise WiringError("cannot record connections after setup failure; construct a new tree")
        if self._phase != "new":
            raise WiringError("record connections before build")
        fields = []
        for block, name in ((source_block, source_port), (destination_block, destination_port)):
            if not isinstance(block, Block) or (block is not self and block._parent is not self):
                raise WiringError("endpoints must belong to this owner or a registered direct child")
            field = block.schema.get(name) if isinstance(name, str) else None
            if field is None or field.kind not in ("In", "Out"):
                raise WiringError(f"{name!r}: endpoint must name an In or Out declaration")
            fields.append(field)
        source, destination = fields
        boundary_input = source_block is self and source.kind == "In"
        child_output = source_block is not self and source.kind == "Out"
        child_input = destination_block is not self and destination.kind == "In"
        boundary_output = destination_block is self and destination.kind == "Out"
        if not ((boundary_input and child_input) or (child_output and (child_input or boundary_output))):
            raise WiringError(
                "unsupported connection: use boundary In to child In, child Out to child In or boundary Out"
            )
        if destination_port in destination_block._incoming:
            raise WiringError(f"{destination_port}: destination already has a source")
        if destination.kind == "In" and destination_port in destination_block._input_raw:
            raise WiringError(f"{destination_port}: destination already has an external input")
        if destination.kind == "Out" and destination_port in destination_block._supplied:
            raise StorageError(f"{destination_port}: a supplied output cannot be forwarded")
        connection = _Connection(source_block, source_port, destination_block, destination_port)
        records = self._connections + (connection,)
        destination_block._incoming[destination_port] = connection
        self._connections = records

    def _endpoint_source(self, name):
        # Inputs travel up boundary forwarding until an external slot or an
        # output is reached. Outputs then travel down to their direct producer.
        # Never follow the producer's inputs: signal feedback is caller-ordered.
        block = self
        while name in block._incoming:
            edge = block._incoming[name]
            block, name = edge.source_block, edge.source_port
        return block, name

    def _configure_connections(self):
        # Every direct child's declarations are now resolved. Input forwarding
        # records are checked here; canonical input slots are a root operation.
        for edge in self._connections:
            source, destination = edge.source_block, edge.destination_block
            source_shape, source_dtype = source._specs[edge.source_port]
            dest_shape, dest_dtype = destination._specs[edge.destination_port]
            label = f"{source._identity()}.{edge.source_port} -> {destination._identity()}.{edge.destination_port}"
            if source_shape != dest_shape:
                raise ShapeError(f"{label}: connection shapes differ: {source_shape} != {dest_shape}")
            if source_dtype != dest_dtype:
                raise DtypeError(f"{label}: connection dtypes differ: {source_dtype} != {dest_dtype}")
            if destination is self:
                view = source._exports[edge.source_port]
                self._values[edge.destination_port] = view
                self._exports[edge.destination_port] = view

    def _finalize_tree(self):
        nodes = tuple(self._walk())
        bindings = []
        external = {}
        for node, name, source, port in self._input_sources():
            if source.schema[port].kind == "Out":
                view = source._exports[port]
            else:
                key = (id(source), port)
                if key not in external:
                    external[key] = (source, port, [])
                external[key][2].append((node, name))
                view = source._values.get(port, MISSING)
            bindings.append((node, name, view))

        writers = tuple(
            arrays.Writer(
                f"{node._identity()}.{writer.name} ({node.schema[writer.name].kind})", writer.array, writer.supplied
            )
            for node in nodes
            for writer in node._writers
        )
        arrays.validate_writers(writers)
        for source, port, readers in external.values():
            if port in source._input_raw:
                arrays.validate_input_storage(
                    source._input_raw[port], f"{source._identity()}.{port} (external input)", writers
                )

        # All storage checks succeeded. Complete input bindings and cache the
        # readers of each canonical external slot.
        for node in nodes:
            node._missing_inputs.clear()
        for node, name, view in bindings:
            if view is MISSING:
                node._values.pop(name, None)
                node._missing_inputs.add(name)
            else:
                node._values[name] = view
        for source, port, readers in external.values():
            source._input_readers[port] = tuple(readers)
        for node in nodes:
            node._refresh_calls()
        self._tree.writers = writers

    def _refresh_calls(self):
        required = set()
        for call in self._calls.values():
            call.missing = tuple(name for name in call.dependencies if name in self._missing_inputs)
            required.update(call.missing)
        self._run_missing = tuple(sorted(required))

    def _assign(self, field, value):
        name, declaration = field.name, field
        if declaration.kind in ("Param", "Control"):
            if declaration.kind == "Param" and self._config_locked:
                raise ParamError(f"{name}: Param cannot be reassigned")
            if declaration.kind == "Control":
                owner = self
                while owner is not None:
                    if owner._pending:
                        raise ParamError(f"{name}: Control is frozen during a split iteration")
                    owner = owner._parent
            if _declarations.is_array_configuration(field.value_type):
                value = arrays.configuration_array(value, name, readonly=True)
            self._values[name] = value
        elif declaration.kind == "In":
            self._bind_input(name, value)
        elif declaration.kind == "Out" or declaration.axes:
            if name in self._values and value is self._values[name]:
                return
            raise StorageError(f"{name}: write in place; supply Out storage using bind_output before setup")
        else:
            if self._unconfigured:
                raise UnavailablePortError(f"{name}: State is initialized during setup")
            self._values[name] = value

    def _bind_input(self, name, value):
        if name in self._incoming:
            raise WiringError(f"{name}: cannot assign to a connected input")
        # Bound shape/dtype/storage metadata must stay fixed. Refilling data is
        # allowed, so the retained caller object needs no new validation or view.
        if name in self._input_raw and value is self._input_raw[name]:
            return
        label = f"{self._identity()}.{name} (external input)"
        arrays.require_array(value, label)
        if name in self._specs:
            shape, dtype = self._specs[name]
            arrays.validate(value, shape, dtype, label)
        if self._ready:
            arrays.validate_input_storage(value, label, self._tree.writers)
        view = arrays.readonly_view(value)
        # Commit only after all validation and view creation succeed.
        self._input_raw[name] = value
        readers = self._input_readers[name] if self._ready else ((self, name),)
        for reader, port in readers:
            reader._values[port] = view
            reader._missing_inputs.discard(port)
            reader._refresh_calls()

    def bind_output(self, name, array):
        field = self.schema.get(name)
        if field is None or field.kind != "Out":
            raise StorageError(f"{name}: not an output")
        if not self._unconfigured and not self._in_setup_hook:
            raise StorageError(f"{name}: output storage must be supplied before allocation")
        if name in self._incoming:
            raise StorageError(f"{name}: bind storage to the direct producer, not a forwarded output")
        arrays.require_array(array, name)
        self._supplied[name] = array

    def _resolve(self, name, declaration):
        dtype = self.ctx.dtype if declaration.dtype is Sample else declaration.dtype
        if dtype is int:
            dtype = np.int64
        try:
            dtype = np.dtype(dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DtypeError(f"{name}: unsupported dtype") from exc
        if dtype.kind not in "biufc":
            raise DtypeError(f"{name}: array dtype must be numeric")
        shape = []
        for axis in declaration.axes:
            if axis is C:
                size = self.ctx.channels
            elif axis is N:
                size = self.ctx.n
            elif isinstance(axis, _dimensions.Expression):
                size = _dimensions.resolve(
                    axis, self._values, self.ctx.channels, self.ctx.n, f"{self._identity()}.{name}"
                )
            elif isinstance(axis, Axis):
                if not isinstance(axis.name, str):
                    raise SetupError(f"{name}: axis name must be a string")
                attribute = inspect.getattr_static(self, axis.name, MISSING)
                field = self.schema.get(axis.name)
                is_param = field is not None and field.kind == "Param"
                is_property = isinstance(attribute, property)
                is_slot = isinstance(attribute, MemberDescriptorType)
                is_instance_value = field is None and (axis.name in vars(self) or is_slot)
                if not (is_param or is_property or is_instance_value):
                    raise SetupError(f"{name}: unresolved or unsupported axis {axis.name}")
                try:
                    # Bypass __getattr__, including when a property raises
                    # AttributeError internally. That author error must survive.
                    size = object.__getattribute__(self, axis.name)
                except SetupError as exc:
                    if "_context_block" not in vars(exc):
                        raise
                    raise SetupError(f"{name}: unresolved axis {axis.name}: {exc}") from exc
                except UnavailablePortError as exc:
                    raise SetupError(f"{name}: unresolved axis {axis.name}: {exc}") from exc
                except AttributeError as exc:
                    if not is_slot:
                        raise
                    raise SetupError(f"{name}: unresolved axis {axis.name}") from exc
            else:
                size = axis
            if isinstance(size, bool) or not isinstance(size, Integral) or size < 0:
                raise SetupError(f"{name}: invalid axis size {size!r}")
            shape.append(int(size))
        return tuple(shape), dtype

    def setup(self, ctx):
        if not self._in_setup_hook:
            return self._setup(ctx, None)

    def _setup(self, ctx, body):
        if not self._unconfigured or self._tree.failed:
            raise SetupError("setup runs once; construct a new block after failure")
        if self._parent is not None and self._parent._phase != "setting-up":
            raise SetupError("a registered child is set up only within its parent's setup")
        if self._parent is None:
            self._build()
        self._phase = "setting-up"
        try:
            if not isinstance(ctx, Ctx) or ctx.xp is not np:
                raise SetupError("Graphion requires Ctx with xp=numpy")
            self._ctx = ctx
            if body is not None:
                self._in_setup_hook = True
                try:
                    body(self, ctx)
                finally:
                    self._in_setup_hook = False
            if self._tree.failed:
                raise SetupError("a child setup failed; construct a new tree")
            writers = []
            for name, field in self.schema.items():
                declaration = field
                if declaration.kind in ("Param", "Control"):
                    continue
                if declaration.kind == "State" and not declaration.axes:
                    continue
                shape, dtype = self._resolve(name, declaration)
                self._specs[name] = shape, dtype
                if declaration.kind == "In":
                    if name in self._input_raw:
                        arrays.validate(self._input_raw[name], shape, dtype, name)
                    continue
                if declaration.kind == "Out" and name in self._incoming:
                    continue
                supplied = name in self._supplied
                if supplied:
                    array = self._supplied[name]
                    arrays.validate(array, shape, dtype, name, writable=True)
                    arrays.validate_output_layout(array, name)
                else:
                    array = arrays.allocate(shape, dtype)
                self._values[name] = array
                writers.append(arrays.Writer(name, array, supplied))
                if declaration.kind == "Out":
                    self._exports[name] = arrays.readonly_view(array)
            self._writers = tuple(writers)
            self._port_specs = {
                name: PortSpec(*spec) for name, spec in self._specs.items() if self.schema[name].kind in ("In", "Out")
            }
            for child in self._children.values():
                if child._phase == "built":
                    child.setup(ctx)
            self._configure_connections()
            exports = tuple(self._exports[name] for name, field in self.schema.items() if field.kind == "Out")
            self._result = exports[0] if len(exports) == 1 else exports or None
            self._initializing_reset = True
            try:
                self.reset()
            finally:
                self._initializing_reset = False
            if self._tree.failed:
                raise SetupError("a child setup failed; construct a new tree")
            self._phase = "configured"
            self._local_setup_complete = True
            if self._parent is None:
                self._finalize_tree()
                self._tree.ready = True
        except Exception:
            self._phase = "failed"
            self._tree.ready = False
            self._tree.failed = True
            raise

    def reset(self):
        if not self._in_reset_hook:
            return self._reset(None)

    def _reset(self, body):
        if not self._ready and not self._initializing_reset:
            raise SetupError(
                "reset requires successful setup of the whole tree; "
                "parent reset hooks must not call child.reset(), since children are already initialized"
            )
        prepare = not self._initializing_reset and not self._reset_prepared
        if prepare:
            for node in self._walk():
                node._pending = False
                node._reset_prepared = True
        try:
            if not self._initializing_reset:
                for child in self._children.values():
                    child.reset()
            for name, default in self._state_defaults.items():
                if self.schema[name].axes:
                    arrays.initialize(self._values[name], default, name)
                else:
                    self._values[name] = default
            if body is not None:
                self._in_reset_hook = True
                try:
                    body(self)
                finally:
                    self._in_reset_hook = False
            self._requires_reset = False
        except BaseException:
            self._requires_reset = True
            raise
        finally:
            if prepare:
                for node in self._walk():
                    node._reset_prepared = False

    def run(self):
        if not self._ready:
            raise SetupError("processing requires successful setup")
        if self._requires_reset:
            raise ExecutionError(f"{self._identity()}: processing requires successful reset")
        if self._pending:
            raise ExecutionError(f"{self._identity()}: iteration already pending")
        call = self._process_call
        if call is None and self._output_call is None:
            raise ExecutionError(f"{self._identity()}: no processing hook defined")
        if self._run_missing:
            raise UnavailablePortError(f"{self._identity()}: unset inputs: {', '.join(self._run_missing)}")
        try:
            if call is not None:
                if call.hook() is not None:
                    raise ExecutionError(f"{self._identity()}: processing hooks must return None")
            else:
                call = self._output_call
                self._pending = True
                if call.hook() is not None:
                    raise ExecutionError(f"{self._identity()}: processing hooks must return None")
                update = self._update_call
                if update is not None and update.hook() is not None:
                    raise ExecutionError(f"{self._identity()}: processing hooks must return None")
                self._pending = False
        except BaseException:
            self._pending = False
            self._requires_reset = True
            raise
        return self._result

    def run_output(self):
        if not self._ready:
            raise SetupError("processing requires successful setup")
        if self._requires_reset:
            raise ExecutionError(f"{self._identity()}: processing requires successful reset")
        call = self._output_call
        if call is None or self._pending:
            raise ExecutionError(f"{self._identity()}: output requires an idle split block")
        if call.missing:
            raise UnavailablePortError(f"{self._identity()}: unset inputs: {', '.join(call.missing)}")
        self._pending = True
        try:
            if call.hook() is not None:
                raise ExecutionError(f"{self._identity()}: processing hooks must return None")
        except BaseException:
            self._pending = False
            self._requires_reset = True
            raise
        self._pending = self._update_call is not None
        return self._result

    def run_update(self):
        if not self._ready:
            raise SetupError("processing requires successful setup")
        if self._requires_reset:
            raise ExecutionError(f"{self._identity()}: processing requires successful reset")
        call = self._update_call
        if call is None or not self._pending:
            raise ExecutionError(f"{self._identity()}: update requires a pending split iteration")
        if call.missing:
            raise UnavailablePortError(f"{self._identity()}: unset inputs: {', '.join(call.missing)}")
        try:
            if call.hook() is not None:
                raise ExecutionError(f"{self._identity()}: processing hooks must return None")
        except BaseException:
            self._pending = False
            self._requires_reset = True
            raise
        self._pending = False
