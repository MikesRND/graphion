"""The session: one protocol over any adapter, for any number of clients.

Nothing here touches a block. Every operation goes through the adapter
contract, so the same session serves a live model and a recording.

A client keeps its own state and sends it with each request: the probes it
shows, the block it inspects, and the serial of the last snapshot it was
given. Execution state is the adapter's and is shared, so a step taken in one
window is seen by every other.
"""

import time
from datetime import datetime
from pathlib import Path

from spikes.inspection import document, views
from spikes.inspection.recording import Recorder


class Observation:
    """One answer to a client: plain facts in ``header`` and whole frames in ``arrays``.

    ``arrays`` is keyed like the snapshot's frames, so two probes on one
    signal share one array, and one copy on the wire.
    """

    def __init__(self, header, arrays):
        self.header = header
        self.arrays = arrays

    def __getitem__(self, key):
        """Read a header field, so callers can treat an observation as its facts."""
        return self.header[key]

    def array(self, probe):
        """Return the samples a probe id refers to, or None when it has none."""
        key = self.header["frames"].get(probe, {}).get("key")
        return None if key is None else self.arrays[key]


class Session:
    """Answer client requests against one adapter."""

    def __init__(self, adapter, record_root=None):
        self.adapter = adapter
        self.record_root = Path(record_root) if record_root else None
        self.recorder = None
        self._producers = document.producers(adapter.document()["description"])
        self._rate_mark = (time.perf_counter(), 0)
        self._rate = None

    def document(self):
        """Return the model document with the current position."""
        return {**self.adapter.document(), "position": self._position(self.adapter.position())}

    def _position(self, position):
        position = dict(position)
        now = time.perf_counter()
        then, ticks = self._rate_mark
        # A rate over at least a quarter second, so a single poll does not make it jump.
        if now - then >= 0.25:
            self._rate = (position["tick"] - ticks) / (now - then)
            self._rate_mark = (now, position["tick"])
        position["ticks_per_s"] = self._rate
        recorder = self.recorder
        position["recording"] = {"path": str(recorder.directory), "ticks": recorder.ticks} if recorder else None
        return position

    def _key(self, probe):
        """Resolve a probe to the signal it shows: its direct producer's output and a channel."""
        address = document.address(probe["path"], probe["port"])
        return (self._producers.get(address, address), int(probe.get("channel", 0)))

    def observe(self, request):
        """Answer from the current snapshot; None when the client's ``since`` already names it."""
        probes = {probe["id"]: self._key(probe) for probe in request.get("probes", [])}
        inspected = request.get("inspect")
        snapshot = self.adapter.snapshot(probes.values(), [inspected] if inspected else [], request.get("since"))
        if snapshot is None:
            return None
        frames, arrays = {}, {}
        for probe, key in probes.items():
            frame = snapshot.frames.get(key)
            # A port asked for after this snapshot was made is in the next one; say nothing about it yet.
            if frame is None:
                continue
            frames[probe] = frame.meta()
            if frame.data is not None:
                frames[probe]["key"] = name = f"{key[0]}#{key[1]}"
                arrays[name] = frame.data
        header = {
            "serial": snapshot.serial,
            "position": self._position(snapshot.position),
            "frames": frames,
            "live": snapshot.live(inspected) if inspected else None,
            "trace": snapshot.spans,
            "trace_partial": snapshot.partial,
        }
        return Observation(header, arrays)

    def _after(self, request):
        """Observe after an action: the client always gets the new state, whatever it had before."""
        return self.observe({**request, "since": None})

    def advance(self, request):
        """Run to the next stop, then observe."""
        self.adapter.advance(request.get("mode", "tick"), request.get("n", 1))
        return self._after(request)

    def run(self, request):
        """Start or stop free running."""
        self.adapter.run(bool(request.get("on")), request.get("ticks_per_second"))
        return self._after(request)

    def reset(self, request):
        """Reset the model, or rewind a recording."""
        self.adapter.reset()
        return self._after(request)

    def seek(self, request):
        """Move a recording's cursor."""
        self.adapter.seek(request["index"])
        return self._after(request)

    def trace(self, request):
        """Switch call tracing."""
        self.adapter.set_trace(bool(request.get("on")))
        return self._after(request)

    def control(self, request):
        """Write a Control and log it as an input to the stream."""
        self.adapter.write_control(request["path"], request["name"], request["value"])
        if self.recorder:
            self.recorder.note_input(request["path"], request["name"], request["value"])
        return self._after(request)

    def value(self, path, name):
        """Fetch a configuration array on demand."""
        with self.adapter.safe():
            return views.value_payload(self.adapter.read(path, name))

    def reference(self, probe, view):
        """Reduce a probe in NumPy from the same snapshot path the clients use.

        Nothing in the page calls this. The checks compare the page's own
        reductions with it.
        """
        observation = self.observe({"probes": [{**probe, "id": "reference"}]})
        meta = observation["frames"].get("reference", {})
        data = observation.array("reference")
        if data is None:
            return meta
        rate = self.adapter.document()["facts"].get(probe["path"], {}).get("ports", {}).get(probe["port"], {})
        rate = rate.get("rate") or {}
        return {**meta, **views.reference(data, view, rate.get("hz")), "tick": observation["position"]["tick"]}

    def record(self, request):
        """Start or stop recording every completed tick."""
        if "record" not in self.adapter.capabilities:
            raise ValueError("this adapter cannot record")
        with self.adapter.safe():
            if request.get("on") and self.recorder is None:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                directory = Path(
                    request.get("path") or (self.record_root or Path.cwd()) / f"{self.adapter.name}-{stamp}"
                )
                self.recorder = Recorder(directory, self.adapter)
                self.adapter.on_tick = self.recorder.capture
            elif not request.get("on") and self.recorder is not None:
                self.adapter.on_tick = None
                self.recorder.close()
                self.recorder = None
        return self._after(request)
