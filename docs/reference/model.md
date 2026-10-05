# Shared block model

[Programming reference](index.md)

<a id="shared-concepts"></a>

## Shared concepts

[Model Semantics](../model-semantics.md) defines the shared vocabulary and modeling
rules: configuration, State, Ports, Connections, hierarchy, dependencies, execution,
Reset, failure, and descriptions. Its author guidance table helps choose a Block's
form; its acceptance examples give concrete sequences and results.

Connection records describe wiring. The caller supplies a Schedule that satisfies
those rules; a topology alone does not determine execution.

<a id="implementation-profiles"></a>

## Implementation Profiles

An **Implementation Profile** specializes those rules with concrete representations,
execution, storage, lifetime, and lifecycle contracts. The
[NumPy frame Implementation Profile](profile.md) supplies this package's Python API.
Its buffer requirements are not universal requirements on other implementations.

Compare implementations using the semantics document's acceptance examples and
explicit numeric tolerances. Structural descriptions and passing cases do not prove
arbitrary author code correct. The [description schema](introspection.md) records
known interface and dependency facts without running processing hooks.
