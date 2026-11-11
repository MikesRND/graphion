# Errors and validation limits

[Programming reference](index.md)

<a id="errors-and-validation-limits"></a>

## Errors and validation limits

All framework errors derive from `BlockError`:

| Error | Meaning |
|---|---|
| `DeclarationError` | Invalid or conflicting declarations, reserved names, unsupported author constructor, missing/unresolved value annotations, invalid field arguments, invalid processing signatures, dependency decorators or mixed forms, or invalid array-configuration defaults. |
| `ParamError` | Missing or unknown construction arguments, invalid array-configuration values, reassignment of a Param, or assignment to a frozen Control. |
| `SetupError` | Invalid ctx/domain configuration or unsupported namespace, unavailable ctx or attempted ctx reassignment, unresolved child-dependent axis, build of a registered child, repeated/failed setup, or public run/reset before the tree is ready. |
| `RegistrationError` | Invalid, duplicate or declaration-conflicting child names, an already configured child, a second lifecycle parent, a registration cycle, or late registration after build. |
| `WiringError` | Invalid or unregistered endpoints, duplicate/conflicting sources, assignment to a connected destination, late wiring after build, or an input resolving to its own block's output. |
| `StorageError` | Attempted storage replacement, invalid output binding, non-writable or unproven writable storage, rejected supplied-output layout, prohibited overlap, or exhaustion of the overlap-check work limit. |
| `UnavailablePortError` | A required external Input is unassigned, or a declared array has not yet been allocated. |
| `ExecutionError` | Invalid Phase sequence or entry point, missing processing hooks, non-`None` hook return, or processing while Reset is required. |
| `ShapeError`, `DtypeError` | A port or array State value does not match its resolved declaration. |

Assigning a different array with `self.y = result` raises `StorageError`
with guidance to write into `self.y`. Assigning the current buffer object
back is harmless, as required for augmented in-place operations.
Shape/dtype mismatches between configured connection endpoints or supplied
storage produce ShapeError/DtypeError failures.

Numerical exceptions propagate unchanged. A failed run may have advanced
state or partially overwritten outputs; there is no rollback. Every invoked hook failure requires successful Reset before further processing.
A failed Reset also blocks processing. Pre-invocation rejections leave the called
Block unchanged and may be retried. See [execution](execution.md#controls-and-failures).

The framework validates declared array interfaces, bookkeeping and the
storage independence specified in [tree-wide overlap validation](storage-and-lifetime.md#tree-wide-overlap-validation). It does not prove arbitrary
scratch independence, signal correctness, rate semantics, feedback
causality, thread safety or scheduling correctness. Read-only NumPy views
provide accidental-write diagnostics; author convention still governs
access through writable producer attributes or other retained aliases.

Overlap checks and cached read-only exports are required Implementation Profile
behavior, not optional debug features. Optional diagnostics may revalidate metadata;
no per-run writable-flag toggling or overlap checking is required.
