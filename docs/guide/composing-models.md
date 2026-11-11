# Compose a model

A composite groups blocks into a model with its own inputs and outputs. Register
the blocks as children, connect their ports and call them in the required order.
This example connects two Gain blocks from the [quick start](../quickstart.md).

```python
from graphion import Array, Block, In, Out, C, N
# Gain is the class defined in the quick start.


class TwoGains(Block):
    x: Array = In(C, N)
    y: Array = Out(C, N)

    def build(self):
        self.a = self.child("a", Gain(gain=2.0))
        self.b = self.child("b", Gain(gain=0.25))
        self.connect(self, "x", self.a, "x")
        self.connect(self.a, "y", self.b, "x")
        self.connect(self.b, "y", self, "y")

    def process(self):
        self.a.run()
        self.b.run()
```

The first Gain doubles the input; the second multiplies that result by 0.25.
Together they produce half the original input. The connections pass data into
`a`, from `a` to `b`, and from `b` to the composite's output.

The `process()` body calls `a` before `b`, so `b` reads the result that `a` just
produced. Calling `setup(ctx)` on the composite configures both children with
the same context. The composite's output shares `b`'s output storage.

## Build, inspect, then set up

The constructor already ran the `TwoGains.build` author body, so its children and
wiring exist immediately. Calling `model.build()` commits the assembled tree and
runs no author body. A caller may also register children or record connections on
an unbuilt tree before calling `build()`; see the [build lifecycle](../reference/lifecycle.md#build).

```python
model = TwoGains()  # the constructor ran build(): children and wiring exist
model.build()  # validate and fix the diagram; no context needed
diagram = model.describe()  # topology "complete"; shapes symbolic; storage None

model.setup(Ctx(rate=8, n=4, channels=1))  # resolve interfaces, allocate, initialize
model.x = np.array([[2.0, 4.0, 6.0, 8.0]])
result = model.run()  # [[1., 2., 3., 4.]]

auto = TwoGains()
auto.setup(Ctx(rate=8, n=4, channels=1))  # setup builds an unbuilt root first
```

`diagram` shows complete topology alongside unresolved interface dimensions and
dtypes, with `storage: null`. A setup failure after build leaves the diagram's
topology complete; storage is reported only after successful root setup.

Details: [ports](../reference/ports-and-connections.md),
[execution](../reference/execution.md).
Next: [manage data](managing-data.md).
