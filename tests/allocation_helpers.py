"""Allocation regression guards for the reference blocks.

Snapshots detect retained NumPy data; peak tracing also catches large freed
temporaries. This is not an event counter or a proof of zero allocations.
"""

import tracemalloc
from dataclasses import dataclass

import numpy as np

from graphion import Array, Block, C, Ctx, In, N, Out
from graphion.dsp.blocks import FrameDelay, Gain, OnePoleLowpass


class GainChain(Block):
    """Two connected half-gain stages, with both boundary ports forwarded."""

    x: Array = In(C, N)
    y: Array = Out(C, N)

    def build(self):
        self.first = self.child("first", Gain(gain=0.5))
        self.second = self.child("second", Gain(gain=0.5))
        self.connect(self, "x", self.first, "x")
        self.connect(self.first, "y", self.second, "x")
        self.connect(self.second, "y", self, "y")

    def process(self):
        self.first.run()
        self.second.run()


PEAK_ALLOWANCE_BYTES = 32 * 1024
WARMUP_CALLS = 3

# OnePoleLowpass traces a Python loop per sample. More channels and fewer samples
# preserve a large signal buffer without spending the test on loop tracing.
ALLOCATION_WORKLOADS = (
    (Gain, 2, 65536, 20),
    (FrameDelay, 2, 65536, 20),
    (OnePoleLowpass, 64, 1024, 5),
    (GainChain, 2, 65536, 20),
)


@dataclass(frozen=True)
class AllocationSample:
    peak_extra_bytes: int
    retained_numpy_blocks: int
    retained_numpy_bytes: int


def sample_allocations(call, *, repeats=20):
    """Trace only warmed calls, with setup-owned buffers outside the window."""
    if tracemalloc.is_tracing():
        raise RuntimeError("allocation checks require their own tracemalloc session")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    for _ in range(WARMUP_CALLS):
        call()
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        for _ in range(repeats):
            call()
        _, peak = tracemalloc.get_traced_memory()
        # Take the snapshot after reading the peak, to exclude snapshot overhead.
        snapshot = tracemalloc.take_snapshot().filter_traces(
            (tracemalloc.DomainFilter(True, np.lib.tracemalloc_domain),)
        )
        return AllocationSample(
            peak_extra_bytes=max(0, peak - baseline),
            retained_numpy_blocks=len(snapshot.traces),
            retained_numpy_bytes=sum(trace.size for trace in snapshot.traces),
        )
    finally:
        tracemalloc.stop()


def check_storage_regression(call, *, frame_bytes, repeats=20):
    """Reject retained new NumPy data and peaks far below one test frame."""
    if frame_bytes <= PEAK_ALLOWANCE_BYTES:
        raise ValueError("use a frame larger than the peak allowance")
    sample = sample_allocations(call, repeats=repeats)
    if sample.retained_numpy_blocks:
        raise AssertionError(f"new NumPy data survives processing: {sample}")
    if sample.peak_extra_bytes >= PEAK_ALLOWANCE_BYTES:
        raise AssertionError(f"processing exceeded the allocation peak allowance: {sample}")
    return sample


def configured_block(cls, dtype, *, channels=2, n=65536):
    ctx = Ctx(rate=20e6, n=n, channels=channels, dtype=dtype)
    block = cls()
    block.setup(ctx)
    block.x = np.ones((ctx.channels, ctx.n), dtype=dtype)
    return block


def persistent_buffers(block):
    """Inspect public declarations plus the known OnePoleLowpass author workspace."""
    observed, owned, forwarded = {}, {}, []

    def visit(node, path):
        forwarded_names = set()
        for edge in node.connections:
            if edge.destination_block is node and node.schema[edge.destination_port].kind == "Out":
                forwarded_names.add(edge.destination_port)
                destination = getattr(node, edge.destination_port)
                source = getattr(edge.source_block, edge.source_port)
                if not np.shares_memory(destination, source):
                    raise AssertionError("forwarded output allocated separate boundary storage")
                if destination.flags.writeable:
                    raise AssertionError("forwarded output is not read-only")
                forwarded.append(f"{path}.{edge.destination_port}")
        for name, field in node.schema.items():
            if field.kind == "Out" or (field.kind == "State" and field.axes):
                key = f"{path}.{name}"
                observed[key] = getattr(node, name)
                if name not in forwarded_names:
                    owned[key] = observed[key]
        if isinstance(node, OnePoleLowpass):
            # This is the shipped author's explicit workspace, not a runtime
            # implementation field; declare it here so it cannot be overlooked.
            key = f"{path}._work"
            observed[key] = owned[key] = node._work
        for name, child in node.children.items():
            visit(child, path + "/" + name.replace("~", "~0").replace("/", "~1"))

    visit(block, "root")
    return observed, owned, forwarded


def check_block_storage(block, *, repeats):
    before, owned, forwarded = persistent_buffers(block)
    exported = block.run()
    sample = check_storage_regression(block.run, frame_bytes=block.y.nbytes, repeats=repeats)
    after, _, _ = persistent_buffers(block)
    if before.keys() != after.keys() or any(after[name] is not value for name, value in before.items()):
        raise AssertionError("persistent output, State or workspace buffer identity changed")
    if block.run() is not exported:
        raise AssertionError("cached output export identity changed")
    storage = {
        "bytes": sum(array.nbytes for array in owned.values()),
        "buffers": {
            name: {"bytes": array.nbytes, "dtype": array.dtype.str, "shape": list(array.shape)}
            for name, array in owned.items()
        },
        "stable_identities": True,
        "forwarded_outputs_without_boundary_storage": forwarded,
    }
    return sample, storage
