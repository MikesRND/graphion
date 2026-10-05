# Programming reference

Blocks declare configuration, state, and input/output ports. This reference defines
how those declarations behave, how blocks are connected and executed, and how their
storage is owned and reused. It includes the exact lifecycle, validation and error
rules needed to write blocks and implement compatible behavior.

For a first example, start with the [quick start](../quickstart.md). For a guided
introduction, follow the [user guide](../guide/index.md).

| Chapter | Covers |
|---|---|
| [Shared block model](model.md) | Parameters, controls, state, ports, hierarchy and execution ownership |
| [Python and NumPy](profile.md) | Synchronous processing, complete frames and supported operations |
| [Declarations](declarations.md) | Fields, annotations, configuration, State and inheritance |
| [Context and types](context-and-types.md) | Dtypes, axes, resolution and array validation |
| [Lifecycle](lifecycle.md) | Children, setup, finalization, reset and hook ordering |
| [Ports and connections](ports-and-connections.md) | Endpoints, supported wiring and forwarding |
| [Storage and lifetime](storage-and-lifetime.md) | Ownership, borrowing, supplied layouts and overlap checks |
| [Execution](execution.md) | Invocation, manual ordering and storage reuse |
| [Introspection](introspection.md) | Class schemas and the structural description format |
| [Errors](errors.md) | Failure classes, recovery limits and validation boundaries |
| [Worked examples](examples.md) | Stateful processing, composition, rate conversion and feedback |

The shared block model describes concepts that can be implemented in different
technologies. The remaining chapters define the Python/NumPy API. Each rule has one
authoritative location; related chapters link to it where the behavior interacts.
