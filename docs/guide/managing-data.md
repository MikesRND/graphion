# Manage data

Graphion reuses array storage across processing calls. Inputs and returned results
are borrowed: read a result before its producer next runs or resets, and copy it
when you need to keep a snapshot.

```python
import numpy as np

# amp is the configured Gain from the quick start.
input_buffer = np.empty((1, 4), dtype=np.float64)
amp.x = input_buffer
input_buffer.fill(2.0)
result = amp.run()
snapshot = result.copy()
input_buffer.fill(4.0)
amp.run()  # result now exposes the newer output
```

Binding once and refilling avoids a new input view on each frame. Replacing an
input is also supported and validated. Store retained history in State, not in
an old reference to an input. Use `out=self.y` or `np.copyto` to write existing
output storage. Convenient allocating expressions remain valid when their
results are copied into that storage.

An owner can supply output storage with `bind_output()` before setup. It must
have the exact interface and an accepted writable layout, and be disjoint from
other declared writers and external inputs. Keep metadata fixed after binding.

Details: [storage, layout and lifetime](../reference/storage-and-lifetime.md),
[errors](../reference/errors.md). For rate conversion and advanced retention,
see the [worked examples](../reference/examples.md).
