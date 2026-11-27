"""Snapshots: one moment of a target, copied once and shared by every client.

An adapter hands the session a snapshot instead of letting it read the target.
A snapshot is immutable and is replaced by reference, so a slow client can
still be sending one while the next is being made. Each frame carries the
stamp that was true when its bytes were copied.
"""

from dataclasses import dataclass, field

import numpy as np

from spikes.inspection import views

# Samples sent per channel. A longer channel is cut and says so; whole frames
# of any size would make one probe cost megabytes per update.
MAX_SAMPLES = 1 << 16


@dataclass(frozen=True)
class Frame:
    """One channel of one port, or the reason there is no data for it."""

    data: np.ndarray | None = None
    dtype: str | None = None
    shape: tuple = ()
    channel: int = 0
    total: int = 0
    stamp: dict | None = None
    unrun: bool = False
    missing: str | None = None

    def meta(self):
        """Return the frame's facts as plain data, without its samples."""
        if self.missing is not None:
            return {"missing": self.missing}
        out = {"dtype": self.dtype, "shape": list(self.shape), "channel": self.channel, "total": self.total}
        if self.unrun:
            return {**out, "unrun": True}
        return {**out, "count": int(self.data.size), "truncated": self.data.size < self.total, "stamp": self.stamp}


def take(array, channel, stamp, limit=MAX_SAMPLES):
    """Copy one channel of an array into a frame; this is the only copy a probe costs the target."""
    array = np.asarray(array)
    if array.ndim == 2:
        if not 0 <= channel < array.shape[0]:
            return Frame(missing=f"no channel {channel}; the port has {array.shape[0]}")
        row = array[channel]
    else:
        row = array.reshape(-1)
    facts = {"dtype": row.dtype.newbyteorder("<").str, "shape": tuple(array.shape), "channel": channel}
    if stamp is None:
        # Output storage is unspecified before its producer first runs; send nothing rather than garbage.
        return Frame(**facts, total=int(row.size), unrun=True)
    # A plain copy, so the frame never aliases the target's buffer. The wire format is
    # little-endian; converting is left to the rare big-endian array, because asking
    # NumPy for a conversion it does not need is many times slower than a copy.
    data = row[:limit].copy()
    if data.dtype.byteorder == ">":
        data = data.astype(data.dtype.newbyteorder("<"))
    return Frame(data=data, **facts, total=int(row.size), stamp=stamp)


@dataclass(frozen=True)
class Snapshot:
    """Everything a client may ask about one moment.

    ``frames`` is keyed by ``(address, channel)`` and ``values`` by block path;
    ``mark`` identifies the moment, so an adapter can tell whether a snapshot
    it already made is still current, and ``made`` is when it was copied.
    """

    serial: int
    mark: tuple
    made: float
    position: dict
    spans: list
    partial: bool
    frames: dict
    values: dict
    _summaries: dict = field(default_factory=dict, compare=False, repr=False)

    def live(self, path):
        """Summarize one block's copied Control and State values, on the caller's thread."""
        if path not in self._summaries:
            raw = self.values.get(path)
            # Two clients may race here; both compute the same summary, so the last write is harmless.
            self._summaries[path] = (
                None if raw is None else {kind: views.summarize_all(members) for kind, members in raw.items()}
            )
        return self._summaries[path]

    def covers(self, ports, blocks):
        """Say whether this snapshot already holds everything a request wants."""
        return all(key in self.frames for key in ports) and all(path in self.values for path in blocks)
