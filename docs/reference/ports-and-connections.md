# Ports and connections

[Programming reference](index.md)

<a id="port-values"></a>

## Port values and connections

Output attributes such as `block.y` expose array storage, not port handles.
Input attributes such as `block.x` expose cached read-only views of their bound
arrays; an unassigned Input raises `UnavailablePortError`. Wiring uses
explicit `(block, name)` endpoints.

<a id="recording-a-connection"></a>

### Recording a connection

Connections belong to a containing block:

```python
self.connect(source, "source_port", destination, "destination_port")
```

The supported boundary patterns are:

| Source | Destination | Meaning |
|---|---|---|
| This block's In | A direct child's In | Pass a boundary input into the implementation. |
| A direct child's Out | A direct child's In | Connect sibling blocks. |
| A direct child's Out | This block's Out | Expose a child's result at the boundary. |

These rules make hierarchy explicit. A connection across deeper nesting
passes through declared boundary ports. Input forwarding goes down the
lifecycle tree; output forwarding goes up it. Neither operation executes
a producer. Runtime endpoint references can be resolved during setup;
there is no graph traversal to schedule work.

Every child endpoint must already be registered as a direct child of the
containing owner when `connect()` is called. Otherwise it raises `WiringError`.
The child need not have been set up. Boundary endpoints refer to the owner itself.

The owner's immutable `connections` view holds records in recording order.
Each record has `source_block` and `destination_block` attributes referring to
the endpoint Block instances, and `source_port` and `destination_port` attributes
holding declaration-name strings. Record attributes cannot be reassigned.
Ownership is given by the containing view; these runtime records are distinct
from the serialized path records returned by `describe()`.

Each destination has at most one source. A source may feed many
destinations. Connections never sum, concatenate, slice or transform
arrays implicitly; those operations belong in blocks.

A direct input-to-output pass-through is an ordinary block that calls
`np.copyto(self.y, self.x, casting="no")` in `output(self)`. It writes its own output buffer
without allocating another signal array.

Connections are recorded before the lifecycle root builds and are fixed by
[build](lifecycle.md#build). They refer to port declarations, so no arrays or
configured children are needed when recording them. Their resolved shapes and
dtypes are checked during setup.

An external input value and a connection cannot coexist at one destination;
adding a connection to a supplied input raises `WiringError` instead of ignoring
data. A rejected connection does not change existing connection records or bindings.

<a id="forwarded-outputs"></a>

### Forwarded outputs

An output forwarding connection resolves to the direct producer's cached
read-only export during setup. The boundary allocates no additional data
buffer or view, and execution does not copy or republish a result.

The forwarding record supplies both the runtime connection and the
diagram edge. No assignment or copy is needed in the owner's processing hook.

Forwarding does not process the source. The owner must have scheduled the
work it needs. A forwarded output changes when its actual producer writes
the buffer, including if that child is run directly. There is no
producer-freshness or execution-order check.
