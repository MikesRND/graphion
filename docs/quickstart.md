# Quick start

Write and run a Gain block that multiplies an input array by a configurable
value. Declare its inputs, outputs and control, then implement the calculation
in `output()`.

```python
import numpy as np
from graphion import Array, Block, Control, Ctx, In, Out, C, N


class Gain(Block):
    gain: float = Control(default=1.0)
    x: Array = In(C, N)
    y: Array = Out(C, N)

    def output(self):
        np.multiply(self.x, self.gain, out=self.y)


amp = Gain(gain=0.5)
amp.setup(Ctx(rate=1_000_000, channels=1, n=4))
amp.x = np.array([[2.0, 4.0, 6.0, 8.0]])
result = amp.run()  # [[1., 2., 3., 4.]]
saved = result.copy()  # retain an independent result
```

Declare the interface, write `output()`, configure once, bind input and call
`run()`. The framework provides output storage. Results borrow it until the
producer next executes or resets. Change Controls between calls.

Continue with the [user guide](guide/index.md); consult the
[programming reference](reference/index.md) for the complete contracts.
