"""The session: one protocol over any adapter.

Nothing here touches a block. Every operation goes through the adapter
contract, so the same session serves a live model and a recording.
"""

import time
from datetime import datetime
from pathlib import Path

from spikes.inspection import document, views
from spikes.inspection.recording import Recorder


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
        return {**self.adapter.document(), "position": self._position()}

    def _position(self):
        position = self.adapter.position()
        now = time.perf_counter()
        then, ticks = self._rate_mark
        # A rate over at least a quarter second, so a single poll does not make it jump.
        if now - then >= 0.25:
            self._rate = (position["tick"] - ticks) / (now - then)
            self._rate_mark = (now, position["tick"])
        position["ticks_per_s"] = self._rate
        position["recording"] = (
            {"path": str(self.recorder.directory), "ticks": self.recorder.ticks} if self.recorder else None
        )
        return position

    def _port_rate(self, path, port):
        facts = self.adapter.document()["facts"].get(path, {}).get("ports", {}).get(port, {})
        rate = facts.get("rate")
        return rate["hz"] if rate else None

    def _probe(self, item):
        path, port = item["path"], item["port"]
        view = item.get("view", "time")
        limit = max(8, min(int(item.get("max", 512)), 4096))
        producer_path, _ = document.split_address(self._producers.get(document.address(path, port), ""))
        try:
            frame = self.adapter.read(path, port)
        except KeyError as missing:
            return {"view": view, "missing": str(missing.args[0]) if missing.args else "not recorded"}
        stamp = self.adapter.stamp(producer_path or path)
        if stamp is None:
            # Output storage is unspecified before its producer first runs; show nothing rather than garbage.
            empty = {"nonfinite": 0, "peak": None, "rms": None}
            return {"view": view, **views.describe_frame(frame), "stats": empty, "unrun": True}
        return {**views.reduce(frame, view, limit, self._port_rate(path, port)), "stamp": stamp}

    def observe(self, request):
        """Answer the probes and the inspected block at the adapter's current safe point."""
        with self.adapter.safe():
            probes = {item["id"]: self._probe(item) for item in request.get("probes", [])}
            inspected = request.get("inspect")
            live = self.adapter.live(inspected) if inspected else None
            spans, partial = self.adapter.spans()
            position = self._position()
        return {"position": position, "probes": probes, "live": live, "trace": spans, "trace_partial": partial}

    def advance(self, request):
        """Run to the next stop, then observe."""
        self.adapter.advance(request.get("mode", "tick"), request.get("n", 1))
        return self.observe(request)

    def run(self, request):
        """Start or stop free running."""
        self.adapter.run(bool(request.get("on")), request.get("ticks_per_second"))
        return self.observe(request)

    def reset(self, request):
        """Reset the model, or rewind a recording."""
        self.adapter.reset()
        return self.observe(request)

    def seek(self, request):
        """Move a recording's cursor."""
        self.adapter.seek(request["index"])
        return self.observe(request)

    def trace(self, request):
        """Switch call tracing."""
        self.adapter.set_trace(bool(request.get("on")))
        return self.observe(request)

    def control(self, request):
        """Write a Control and log it as an input to the stream."""
        self.adapter.write_control(request["path"], request["name"], request["value"])
        if self.recorder:
            self.recorder.note_input(request["path"], request["name"], request["value"])
        return self.observe(request)

    def value(self, path, name):
        """Fetch a configuration array on demand."""
        with self.adapter.safe():
            return views.value_payload(self.adapter.read(path, name))

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
        return self.observe(request)
