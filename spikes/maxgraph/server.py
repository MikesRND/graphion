"""Session server for the maxGraph canvas spike.

Serves a structural description plus the facts the diagram needed and
describe() does not carry, and advances the model one root iteration per
request. Standard library and NumPy only. Not a public Graphion API.
"""

import argparse
import importlib
import inspect
import json
import os
import sys
import time
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, quote, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".map": "application/json"}


def build_model(name):
    """Build and set up one of the example testbenches; return (root, tick callable)."""
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


def extras(root):
    """Facts the UI wanted beyond schema v2: source, docstring, per-port rate, buffer bytes."""
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
        out[path] = {"source": _source(cls), "doc": doc.splitlines()[0] if doc else None, "ports": ports}
    return out


class Tracer:
    """Record run-entry calls with sys.monitoring, leaving the framework's run path untouched."""

    def __init__(self, root):
        from graphion.profile.numpy_frame.block import Block

        self.paths = {id(block): path for path, block in walk(root)}
        self.entries = {
            Block.run.__code__: "run",
            Block.run_output.__code__: "run_output",
            Block.run_update.__code__: "run_update",
        }
        self.records = []
        self.stack = []
        self.origin = 0
        monitoring = sys.monitoring
        self.tool = monitoring.PROFILER_ID
        monitoring.use_tool_id(self.tool, "graphion-maxgraph-spike")
        events = monitoring.events
        monitoring.register_callback(self.tool, events.PY_START, self._start)
        monitoring.register_callback(self.tool, events.PY_RETURN, self._return)
        for code in self.entries:
            monitoring.set_local_events(self.tool, code, events.PY_START | events.PY_RETURN)

    def _start(self, code, offset):
        block = sys._getframe(1).f_locals.get("self")
        record = {
            "path": self.paths.get(id(block), "?"),
            "entry": self.entries[code],
            "depth": len(self.stack),
            "order": len(self.records),
        }
        now = time.perf_counter_ns()
        if not self.records:
            self.origin = now
        record["t0_us"] = round((now - self.origin) / 1000, 1)
        self.records.append(record)
        self.stack.append((record, now, [0]))

    def _return(self, code, offset, retval):
        if not self.stack:
            return
        record, started, children = self.stack.pop()
        elapsed = time.perf_counter_ns() - started
        record["us"] = round(elapsed / 1000, 1)
        record["self_us"] = round((elapsed - children[0]) / 1000, 1)
        if self.stack:
            self.stack[-1][2][0] += elapsed

    def take(self):
        """Return and clear the records of the calls made since the last take."""
        records, self.records, self.stack = self.records, [], []
        return records


def _numbers(array, digits=4):
    return np.round(np.asarray(array, dtype=np.float64), digits).tolist()


def _stats(array):
    data = np.asarray(array)
    magnitude = np.abs(data).astype(np.float64)
    finite = np.isfinite(magnitude)
    if not finite.any():
        return {"nonfinite": int(magnitude.size), "peak": None, "rms": None}
    peak = float(magnitude[finite].max())
    # Scale by the peak so squaring very large values cannot overflow to infinity.
    rms = peak * float(np.sqrt(np.mean((magnitude[finite] / peak) ** 2))) if peak else 0.0
    return {"nonfinite": int(magnitude.size - np.count_nonzero(finite)), "peak": peak, "rms": rms}


def _pool_max(values, limit):
    """Decimate for display by keeping the maximum of each bucket."""
    if values.size <= limit:
        return values
    size = int(np.ceil(values.size / limit))
    padded = np.full(size * int(np.ceil(values.size / size)), -np.inf)
    padded[: values.size] = values
    return padded.reshape(-1, size).max(axis=1)


def probe(block, port, view, limit, rate, fresh):
    """Copy one port's current frame into a display-sized payload."""
    frame = np.asarray(getattr(block, port))
    if not fresh:
        # Output storage is unspecified before the first run and after reset; show nothing rather than garbage.
        empty = {"nonfinite": 0, "peak": None, "rms": None}
        return {"view": view, "dtype": frame.dtype.name, "shape": list(frame.shape), "stats": empty, "unrun": True}
    data = frame[0] if frame.ndim == 2 else frame.reshape(-1)
    out = {"view": view, "dtype": frame.dtype.name, "shape": list(frame.shape), "stats": _stats(frame)}
    is_complex = np.iscomplexobj(data)
    if view == "constellation" and is_complex:
        tail = data[-limit:]
        tail = tail[np.isfinite(tail)]
        out.update(re=_numbers(tail.real), im=_numbers(tail.imag))
    elif view == "spectrum" and data.size >= 8 and data.dtype.kind in "fc":
        window = np.hanning(data.size)
        spectrum = np.fft.fft(data * window) if is_complex else np.fft.rfft(data * window)
        if is_complex:
            spectrum = np.fft.fftshift(spectrum)
        power = np.abs(spectrum) ** 2 / np.sum(window**2)
        db = 10 * np.log10(np.maximum(_pool_max(power, limit), 1e-20))
        half = (rate or 1.0) / 2
        out.update(db=_numbers(db, 2), f0=-half if is_complex else 0.0, f1=half, has_rate=rate is not None)
    elif view == "array" and not is_complex:
        # A whole-frame view for outputs that are already a display, such as a PSD.
        out.update(y=_numbers(_pool_max(data.astype(np.float64), limit)))
    else:
        head = data[:limit]
        out["view"] = "time"
        if is_complex:
            out.update(re=_numbers(head.real), im=_numbers(head.imag))
        else:
            out.update(y=_numbers(head))
    return out


def _live_value(value):
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        head = flat[:8]
        text = ", ".join(f"{v:.4g}" for v in (head if not np.iscomplexobj(head) else np.abs(head)))
        return {"kind": "array", "dtype": value.dtype.name, "shape": list(value.shape), "head": text, **_stats(value)}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return {"kind": "scalar", "value": value}
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return {"kind": "scalar", "value": value.item()}
    return {"kind": "opaque", "type": type(value).__name__}


def live_block(block):
    """Return one block's current Control and State values, read between iterations."""
    out = {"controls": {}, "state": {}}
    for name, record in type(block).schema.items():
        if record.kind == "Control":
            out["controls"][name] = _live_value(getattr(block, name))
        elif record.kind == "State":
            out["state"][name] = _live_value(getattr(block, name))
    return out


def fetch_value(block, name):
    """Lazily fetch a configuration array, with a frequency response when it looks like taps."""
    value = np.asarray(getattr(block, name))
    if value.dtype.kind not in "biufc" or value.size > 65536:
        return {"error": "not a displayable numeric array"}
    flat = value.reshape(-1)
    out = {"dtype": value.dtype.name, "shape": list(value.shape)}
    if np.iscomplexobj(flat):
        out.update(re=_numbers(flat.real, 6), im=_numbers(flat.imag, 6))
    else:
        out["y"] = _numbers(flat, 6)
    if value.ndim == 1 and flat.size >= 8 and value.dtype.kind in "fc":
        response = np.abs(np.fft.fft(flat, 1024))
        if not np.iscomplexobj(flat):
            response = response[:513]
        peak = float(response.max()) or 1.0
        out["response_db"] = _numbers(20 * np.log10(np.maximum(response / peak, 1e-8)), 2)
    return out


class Session:
    """One model, its tracer and a tick counter, guarded for the threaded server."""

    def __init__(self, name):
        self.name = name
        self.lock = Lock()
        self.root = build_model(name)
        self.blocks = dict(walk(self.root))
        self.tracer = Tracer(self.root)
        self.ticks = 0
        self.extras = extras(self.root)

    def model(self):
        """Return the description plus the extra facts, as one JSON-ready mapping."""
        return {"model": self.name, "description": self.root.describe(), "extras": self.extras, "ticks": self.ticks}

    def _rate(self, path, port):
        rate = self.extras[path]["ports"].get(port, {}).get("rate")
        return rate["hz"] if rate else None

    def observe(self, request):
        """Answer the probes and the inspected block for the current model state."""
        probes = {}
        for item in request.get("probes", []):
            block = self.blocks[item["path"]]
            limit = max(8, min(int(item.get("max", 512)), 4096))
            rate = self._rate(item["path"], item["port"])
            probes[item["id"]] = probe(block, item["port"], item.get("view", "time"), limit, rate, self.ticks > 0)
        inspected = request.get("inspect")
        live = live_block(self.blocks[inspected]) if inspected in self.blocks else None
        return {"ticks": self.ticks, "probes": probes, "live": live}

    def tick(self, request):
        """Run the root for the requested iterations; the root's run() is the tick."""
        count = max(0, min(int(request.get("n", 1)), 1000))
        self.tracer.take()
        started = time.perf_counter_ns()
        trace = []
        for _ in range(count):
            self.root.run()
            trace = self.tracer.take()
        elapsed = (time.perf_counter_ns() - started) / 1e3
        self.ticks += count
        return {**self.observe(request), "trace": trace, "us_per_tick": round(elapsed / count, 1) if count else None}

    def reset(self, request):
        """Reset the root and report the cleared state."""
        self.root.reset()
        self.ticks = 0
        return self.observe(request)

    def control(self, request):
        """Assign a Control between iterations; framework errors come back as text."""
        block = self.blocks[request["path"]]
        if type(block).schema[request["name"]].kind != "Control":
            raise ValueError("only Controls can change while running")
        current = getattr(block, request["name"])
        setattr(
            block,
            request["name"],
            type(current)(request["value"]) if isinstance(current, (int, float)) else request["value"],
        )
        return {"live": live_block(block)}


class Handler(BaseHTTPRequestHandler):
    """Static files for the page and a small JSON API over the session."""

    session: Session

    def log_message(self, format, *args):
        """Keep the console quiet; ticks arrive many times a second."""

    def _json(self, payload, status=200):
        body = json.dumps(payload, allow_nan=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _guarded(self, call):
        try:
            with self.session.lock:
                self._json(call())
        except Exception as error:  # A spike reports every failure to the page rather than dropping the request.
            self._json({"error": f"{type(error).__name__}: {error}"}, 400)

    def do_GET(self):
        """Serve the page, its bundle and read-only queries."""
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        if url.path == "/api/model":
            return self._guarded(self.session.model)
        if url.path == "/api/value":
            return self._guarded(lambda: fetch_value(self.session.blocks[query["path"]], query["name"]))
        relative = "index.html" if url.path == "/" else url.path.lstrip("/")
        target = (HERE / relative).resolve()
        if not target.is_relative_to(HERE) or not target.is_file() or target.suffix not in CONTENT_TYPES:
            return self._json({"error": "not found"}, 404)
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES[target.suffix])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        return None

    def do_POST(self):
        """Advance, reset, observe or reconfigure the model."""
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        actions = {
            "/api/tick": self.session.tick,
            "/api/observe": self.session.observe,
            "/api/reset": self.session.reset,
            "/api/control": self.session.control,
        }
        action = actions.get(urlparse(self.path).path)
        if action is None:
            return self._json({"error": "not found"}, 404)
        return self._guarded(lambda: action(request))


def main():
    """Parse arguments and serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("ber", "ddc"), default="ber")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    sys.path.insert(0, str(REPO))
    importlib.invalidate_caches()
    Handler.session = Session(args.model)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Graphion maxGraph spike · model {args.model} · http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
