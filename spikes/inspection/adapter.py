"""The live adapter for the Python/NumPy frame Implementation Profile.

An adapter is the only code that touches block objects. It runs the model on
its own thread, lets readers in only at safe points (between calls), and
observes run entry points from outside with sys.monitoring, so the framework's
run path is unchanged and costs nothing extra while tracing is off.

Adapter contract used by the session (the replay adapter implements the same):

    capabilities, document(), position(), safe(), read(path, name),
    live(path), stamp(path), spans(), advance(mode, n), reset(),
    plus the optional run(), set_trace(), write_control(), seek().
"""

import inspect
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import numpy as np

from spikes.inspection import document, views

REPO = Path(__file__).resolve().parents[2]


class Unsupported(Exception):
    """The adapter does not offer the requested operation."""


def build_model(name):
    """Build and set up one of the example testbenches."""
    from fractions import Fraction

    from graphion import Ctx

    if name == "ber":
        from examples.ber.ber_tb import BerTestbench
        from graphion.dsp.communications import qpsk

        root = BerTestbench(constellation=qpsk(), ebn0_db=8.0, viewer_depth=1024)
        root.setup(Ctx(n=4096, channels=1, rate=Fraction(2_000_000)))
    elif name == "ddc":
        from examples.ddc.ddc_tb import ADC_RATE, Testbench

        root = Testbench()
        root.setup(Ctx(rate=ADC_RATE, n=4096, channels=1))
    else:
        raise ValueError(f"unknown model {name!r}")
    return root


def walk(block, path="root"):
    """Yield (path, block) in the same preorder and path spelling describe() uses."""
    yield path, block
    for name, child in block.children.items():
        yield from walk(child, f"{path}/{name.replace('~', '~0').replace('/', '~1')}")


def _source(cls):
    try:
        file = inspect.getsourcefile(cls)
        line = inspect.getsourcelines(cls)[1]
    except (OSError, TypeError):
        return None
    if file is None:
        return None
    path = Path(file).resolve()
    # VS Code opens file:line from a URI; under WSL the file lives behind the remote authority.
    distro = os.environ.get("WSL_DISTRO_NAME")
    authority = f"vscode-remote/wsl+{quote(distro)}" if distro else "file"
    return {
        "text": f"{path.relative_to(REPO) if path.is_relative_to(REPO) else path}:{line}",
        "uri": f"vscode://{authority}{quote(path.as_posix())}:{line}",
    }


def _port_rate(block, port):
    """Per-port rate through the DSP stream convention; None when the port is outside it."""
    from graphion import SetupError
    from graphion.dsp.streams import stream_ctx

    try:
        rate = stream_ctx(block, port).rate
    except SetupError:
        return None
    return {"text": str(rate), "hz": float(rate)}


def gather_facts(root):
    """Collect the provider facts for every block of a set-up model."""
    out = {}
    for path, block in walk(root):
        cls = type(block)
        ports = {}
        for name, record in cls.schema.items():
            if record.axes is None:
                continue
            entry = {"nbytes": int(np.asarray(getattr(block, name)).nbytes)}
            if record.kind in ("In", "Out"):
                entry["rate"] = _port_rate(block, name)
            ports[name] = entry
        # Own docstring only: an inherited Block docstring says nothing about this definition.
        doc = inspect.cleandoc(cls.__doc__) if cls.__dict__.get("__doc__") else None
        out[path] = {
            "source": _source(cls),
            "doc": doc.splitlines()[0] if doc else None,
            "display": document.DISPLAY_HINTS.get((cls.__module__, cls.__qualname__)),
            "ports": ports,
        }
    return out


class NumpyFrameAdapter:
    """Drive and observe one set-up Python/NumPy frame model."""

    capabilities = frozenset(
        {
            "execution.tick",
            "execution.free_run",
            "execution.call_step",
            "trace.calls",
            "trace.toggle",
            "values.write",
            "values.fetch",
            "record",
        }
    )

    def __init__(self, root, name="model"):
        from graphion.profile.numpy_frame.block import Block

        self.root = root
        self.name = name
        self.blocks = dict(walk(root))
        self._paths = {id(block): path for path, block in self.blocks.items()}
        self._document = document.compose(root.describe(), gather_facts(root), self.capabilities, name)

        # One condition guards every field below. The model thread holds "busy"
        # while hooks run; readers wait for it to clear and take priority over the next tick.
        self._cv = threading.Condition()
        self._busy = False
        self._waiters = 0
        self._model_turn = False
        self._closed = False
        self._pending = 0
        self._free = False
        self._period = 0.0
        self._resume = False
        self._stops = 0
        self._state = "idle"
        self._at = None
        self._error = None
        self._ticks = 0
        self.on_tick = None

        self._trace = False
        self._step_calls = False
        self._spans = []
        self._spans_last = []
        self._stack = []
        self._origin = 0
        self._paused_ns = 0
        self._produced = {}
        self._trace_since = 0

        monitoring = sys.monitoring
        self._tool = monitoring.PROFILER_ID
        # A second adapter in one process takes over the tool id the first one registered.
        if monitoring.get_tool(self._tool) is None:
            monitoring.use_tool_id(self._tool, "graphion-inspection-spike")
        self._entries = {
            Block.run.__code__: "run",
            Block.run_output.__code__: "run_output",
            Block.run_update.__code__: "run_update",
        }
        monitoring.register_callback(self._tool, monitoring.events.PY_START, self._call_start)
        monitoring.register_callback(self._tool, monitoring.events.PY_RETURN, self._call_return)

        self._thread = threading.Thread(target=self._loop, name="graphion-model", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ model thread

    def _loop(self):
        while True:
            with self._cv:
                self._cv.wait_for(
                    lambda: self._closed or (self._has_work() and self._may_run() and self._state != "faulted")
                )
                if self._closed:
                    return
                self._model_turn = False
                if self._pending > 0:
                    self._pending -= 1
                self._busy = True
                self._state = "running"
            started = time.perf_counter()
            error = None
            try:
                self._run_tick()
            except Exception as failure:  # The model's own failure is reported as session state, not raised here.
                error = f"{type(failure).__name__}: {failure}"
            with self._cv:
                self._busy = False
                self._at = None
                if error is None:
                    self._ticks += 1
                    self._state = "running" if (self._free or self._pending) else "idle"
                else:
                    self._state, self._error = "faulted", error
                    self._free, self._pending = False, 0
                if self._state != "running":
                    self._stops += 1
                self._cv.notify_all()
                period = self._period if self._free else 0.0
            if period:
                time.sleep(max(0.0, period - (time.perf_counter() - started)))

    def _has_work(self):
        return self._pending > 0 or self._free or self._resume

    def _may_run(self):
        """Readers go first, but after each read the model is owed one step so polling cannot starve it."""
        return self._waiters == 0 or self._model_turn

    def _run_tick(self):
        self._spans = []
        self._stack = []
        self._paused_ns = 0
        self.root.run()
        if self._trace:
            self._spans_last = self._spans
        if self.on_tick is not None:
            # Still "busy": this is a safe point on the model thread, before readers are let in.
            self.on_tick(self._ticks + 1)

    def _clock(self):
        """Model time in nanoseconds, excluding time spent paused at a call."""
        return time.perf_counter_ns() - self._paused_ns

    def _call_start(self, code, offset):
        block = sys._getframe(1).f_locals.get("self")
        path = self._paths.get(id(block), "?")
        entry = self._entries[code]
        # The root's own call is the tick itself; pausing there would stop before anything ran.
        if self._step_calls and self._stack:
            self._pause(path, entry)
        now = self._clock()
        if not self._spans:
            self._origin = now
        span = {
            "path": path,
            "entry": entry,
            "depth": len(self._stack),
            "order": len(self._spans),
            "t0_us": round((now - self._origin) / 1000, 1),
        }
        self._spans.append(span)
        self._stack.append((span, now, [0]))

    def _call_return(self, code, offset, retval):
        if not self._stack:
            return
        span, started, children = self._stack.pop()
        elapsed = self._clock() - started
        span["us"] = round(elapsed / 1000, 1)
        span["self_us"] = round((elapsed - children[0]) / 1000, 1)
        if self._stack:
            self._stack[-1][2][0] += elapsed
        self._produced[span["path"]] = self._ticks + 1

    def _pause(self, path, entry):
        """Block the model thread before a call until a client resumes it."""
        started = time.perf_counter_ns()
        with self._cv:
            self._busy = False
            self._state = "paused"
            self._at = {"path": path, "entry": entry}
            self._stops += 1
            self._cv.notify_all()
            self._cv.wait_for(lambda: self._resume and self._may_run())
            self._model_turn = False
            self._resume = False
            self._busy = True
            self._state = "running"
            self._at = None
        self._paused_ns += time.perf_counter_ns() - started

    # ---------------------------------------------------------------- reading

    @contextmanager
    def safe(self):
        """Hold the model at a safe point (between calls) for the duration of a read."""
        with self._cv:
            self._waiters += 1
            try:
                self._cv.wait_for(lambda: not self._busy and not (self._model_turn and self._has_work()))
                yield
            finally:
                self._waiters -= 1
                self._model_turn = True
                self._cv.notify_all()

    def document(self):
        """Return the composed model document."""
        return self._document

    def position(self):
        """Return where execution stands."""
        return {
            "tick": self._ticks,
            "state": self._state,
            "at": self._at,
            "error": self._error,
            "trace": self._trace,
            "total": None,
        }

    def read(self, path, name):
        """Return the live array or value of a named member; call inside safe()."""
        return getattr(self.blocks[path], name)

    def live(self, path):
        """Return current Control and State values of one block; call inside safe()."""
        block = self.blocks.get(path)
        out = {"controls": {}, "state": {}}
        if block is None:
            return out
        for name, record in type(block).schema.items():
            if record.kind == "Control":
                out["controls"][name] = views.summarize(getattr(block, name))
            elif record.kind == "State":
                out["state"][name] = views.summarize(getattr(block, name))
        return out

    def stamp(self, path):
        """Say which tick last produced a block's outputs, and how that is known.

        With tracing on the answer is observed per block. With tracing off it is
        assumed from the tick count, which holds only if every block ran each tick.
        """
        if path in self._produced:
            return {"tick": self._produced[path], "basis": "observed"}
        # Before a block has been seen running, fall back to the last tick nobody was watching.
        assumed = self._trace_since if self._trace else self._ticks
        return {"tick": assumed, "basis": "assumed"} if assumed > 0 else None

    def spans(self):
        """Return call spans: the tick in progress when paused, else the last complete tick."""
        if self._state == "paused":
            return [dict(span) for span in self._spans], True
        return self._spans_last, False

    # ------------------------------------------------------------- execution

    def _wait_for_stop(self, before):
        self._cv.wait_for(lambda: self._stops != before)

    def advance(self, mode="tick", n=1):
        """Run to the next stop: the end of n ticks, or the next call boundary."""
        with self._cv:
            if self._state == "faulted":
                return
            if mode == "call" and not self._trace:
                raise Unsupported("call stepping needs call tracing switched on")
            self._free = False
            self._step_calls = mode == "call"
            before = self._stops
            if self._state == "paused":
                self._resume = True
            else:
                self._pending = 1 if mode == "call" else max(1, int(n))
            self._cv.notify_all()
            self._wait_for_stop(before)

    def run(self, on, ticks_per_second=None):
        """Start or stop free running; the model does not wait for any client."""
        with self._cv:
            self._free = bool(on) and self._state != "faulted"
            self._period = 1.0 / ticks_per_second if ticks_per_second else 0.0
            self._step_calls = False
            if self._state == "paused":
                self._resume = True
            self._cv.notify_all()
            if not on:
                self._cv.wait_for(lambda: not self._busy and self._state != "paused")
                # Stopped between ticks: the loop is idle but has not had a turn to say so.
                if self._state == "running":
                    self._state = "idle"

    def reset(self):
        """Reset the model. A tick paused part-way is finished first; hooks must not be abandoned mid-call."""
        with self._cv:
            self._free = False
            self._step_calls = False
            self._pending = 0
            if self._state == "paused":
                before = self._stops
                self._resume = True
                self._cv.notify_all()
                self._wait_for_stop(before)
            self._cv.wait_for(lambda: not self._busy)
            self.root.reset()
            self._ticks = 0
            self._trace_since = 0
            self._produced.clear()
            self._spans_last = []
            self._state, self._error, self._at = "idle", None, None
            self._cv.notify_all()

    def set_trace(self, on):
        """Switch call tracing on or off. Off removes the instrumentation entirely."""
        monitoring = sys.monitoring
        events = monitoring.events.PY_START | monitoring.events.PY_RETURN if on else monitoring.events.NO_EVENTS
        with self.safe():
            for code in self._entries:
                monitoring.set_local_events(self._tool, code, events)
            self._trace = bool(on)
            self._trace_since = self._ticks
            if not on:
                # Without tracing nothing keeps these current, so drop them rather than show stale facts.
                self._step_calls = False
                self._produced.clear()
                self._spans_last = []

    def write_control(self, path, name, value):
        """Assign a Control between calls; the framework's own errors propagate."""
        block = self.blocks[path]
        if type(block).schema[name].kind != "Control":
            raise Unsupported("only Controls can change while running")
        with self.safe():
            current = getattr(block, name)
            setattr(block, name, type(current)(value) if isinstance(current, (int, float)) else value)

    def seek(self, tick):
        """Live models cannot move in time."""
        raise Unsupported("a live model cannot seek")

    def close(self):
        """Stop the model thread and remove the instrumentation."""
        self.run(False)
        self.set_trace(False)
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._thread.join(timeout=2)
