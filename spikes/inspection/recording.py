"""Recording a session's stream, and an adapter that replays it with no model.

A recording is a directory:

    manifest.json   the model document and the configuration arrays
    ticks.jsonl     one line per completed tick: spans, live values, frame index
    frames.bin      raw array bytes, appended in order

Frames are full copies of every direct producer's output, so any view can be
recomputed on replay. The replay adapter answers the same calls as a live one;
the session and the user interface cannot tell them apart except by capability.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from spikes.inspection import document
from spikes.inspection.adapter import Unsupported

FORMAT = "graphion.inspection.recording/0"


class Recorder:
    """Append one record per completed tick; called on the model thread at a safe point."""

    def __init__(self, directory, adapter):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.adapter = adapter
        self.ticks = 0
        self._inputs = []
        self._frames = (self.directory / "frames.bin").open("wb")
        self._lines = (self.directory / "ticks.jsonl").open("w")
        self._offset = 0
        doc = adapter.document()
        # Record each signal once, at the output that computes it.
        outputs = {
            document.address(block["path"], declaration["name"])
            for block in doc["description"]["blocks"]
            for declaration in block["declarations"]
            if declaration["kind"] == "Out"
        }
        producers = document.producers(doc["description"])
        self._addresses = sorted({producer for producer in producers.values() if producer in outputs})
        values = {}
        for block in doc["description"]["blocks"]:
            for name, value in block["parameters"].items():
                if value["kind"] == "summary":
                    values[document.address(block["path"], name)] = self._write(adapter.read(block["path"], name))
        manifest = {"format": FORMAT, "document": doc, "values": values}
        (self.directory / "manifest.json").write_text(json.dumps(manifest, default=str))

    def _write(self, array):
        data = np.ascontiguousarray(array)
        entry = [self._offset, data.dtype.str, list(data.shape)]
        self._frames.write(data.tobytes())
        self._offset += data.nbytes
        return entry

    def note_input(self, path, name, value):
        """Log a Control write so a replay shows what was changed and when."""
        self._inputs.append({"path": path, "name": name, "value": value})

    def capture(self, tick):
        """Copy this tick's outputs, spans and live values into the recording."""
        adapter = self.adapter
        frames = {address: self._write(adapter.read(*document.split_address(address))) for address in self._addresses}
        record = {
            "tick": tick,
            "spans": adapter.spans()[0],
            "live": {path: adapter.live(path) for path in adapter.blocks},
            "frames": frames,
            "inputs": self._inputs,
        }
        self._inputs = []
        self._lines.write(json.dumps(record, allow_nan=False) + "\n")
        self.ticks += 1

    def close(self):
        """Flush and close the recording's files."""
        self._frames.close()
        self._lines.close()


class ReplayAdapter:
    """Serve a recording through the adapter contract; no model is loaded."""

    capabilities = frozenset({"execution.tick", "execution.seek", "trace.calls", "values.fetch"})

    def __init__(self, directory):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if manifest.get("format") != FORMAT:
            raise ValueError(f"unsupported recording format {manifest.get('format')!r}")
        self._values = manifest["values"]
        self._ticks = [json.loads(line) for line in (self.directory / "ticks.jsonl").read_text().splitlines() if line]
        self._frames = np.memmap(self.directory / "frames.bin", dtype=np.uint8, mode="r")
        self._document = {**manifest["document"], "capabilities": sorted(self.capabilities)}
        self._producers = document.producers(self._document["description"])
        self.name = self._document["model"]
        self._cursor = 0

    def _record(self):
        return self._ticks[self._cursor - 1] if self._cursor else None

    def _array(self, entry):
        offset, dtype, shape = entry
        count = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        return np.frombuffer(self._frames[offset : offset + count], dtype=dtype).reshape(shape)

    @contextmanager
    def safe(self):
        """Nothing runs during replay, so every moment is a safe point."""
        yield

    def document(self):
        """Return the recorded document with this adapter's capabilities."""
        return self._document

    def position(self):
        """Return the cursor as an execution position."""
        record = self._record()
        return {
            "tick": record["tick"] if record else 0,
            "state": "ended" if self._cursor == len(self._ticks) else "idle",
            "at": None,
            "error": None,
            "trace": bool(record and record["spans"]),
            "total": len(self._ticks),
            "index": self._cursor,
        }

    def read(self, path, name):
        """Return a recorded array; KeyError means it was not recorded."""
        address = document.address(path, name)
        if address in self._values:
            return self._array(self._values[address])
        record = self._record()
        if record is None:
            raise KeyError("before the first recorded tick")
        return self._array(record["frames"][self._producers.get(address, address)])

    def live(self, path):
        """Return the recorded Control and State summaries of one block."""
        record = self._record()
        return record["live"].get(path, {"controls": {}, "state": {}}) if record else {"controls": {}, "state": {}}

    def stamp(self, path):
        """Every recorded frame belongs to a completed tick."""
        record = self._record()
        return {"tick": record["tick"], "basis": "recorded"} if record else None

    def spans(self):
        """Return the recorded spans of the tick under the cursor."""
        record = self._record()
        return (record["spans"] if record else []), False

    def advance(self, mode="tick", n=1):
        """Move the cursor forward; a recording has no call boundaries to stop at."""
        if mode != "tick":
            raise Unsupported("a recording can only step by whole ticks")
        self._cursor = min(len(self._ticks), self._cursor + max(1, int(n)))

    def seek(self, index):
        """Move the cursor to the given recorded tick, counted from one; zero is before the first."""
        self._cursor = max(0, min(len(self._ticks), int(index)))

    def reset(self):
        """Return to the start of the recording."""
        self._cursor = 0

    def run(self, on, ticks_per_second=None):
        """Replay has no thread of its own; the client steps it."""
        raise Unsupported("a recording does not free-run; step or seek it")

    def set_trace(self, on):
        """Spans are either in the recording or not."""
        raise Unsupported("tracing cannot change during replay")

    def write_control(self, path, name, value):
        """Refuse: a recording is read-only."""
        raise Unsupported("a recording is read-only")
