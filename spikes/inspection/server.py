"""HTTP transport for the inspection session, and the spike's command line.

    python -m spikes.inspection.server --model ddc
    python -m spikes.inspection.server --replay spikes/inspection/recordings/<name>

The transport only maps URLs to session calls. Standard library and NumPy only.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from spikes.inspection.session import Session

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "maxgraph"
CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".map": "application/json"}


class Handler(BaseHTTPRequestHandler):
    """Static files for the page and a small JSON API over the session."""

    session: Session

    def log_message(self, format, *args):
        """Keep the console quiet; polls arrive many times a second."""

    def _json(self, payload, status=200):
        body = json.dumps(payload, allow_nan=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _guarded(self, call):
        try:
            self._json(call())
        except Exception as error:  # A spike reports every failure to the page rather than dropping the request.
            self._json({"error": f"{type(error).__name__}: {error}"}, 400)

    def do_GET(self):
        """Serve the page, its bundle and read-only queries."""
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        if url.path == "/api/document":
            return self._guarded(self.session.document)
        if url.path == "/api/value":
            return self._guarded(lambda: self.session.value(query["path"], query["name"]))
        relative = "index.html" if url.path == "/" else url.path.lstrip("/")
        target = (STATIC / relative).resolve()
        if not target.is_relative_to(STATIC) or not target.is_file() or target.suffix not in CONTENT_TYPES:
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
        """Dispatch one session operation."""
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        session = self.session
        actions = {
            "/api/observe": session.observe,
            "/api/advance": session.advance,
            "/api/run": session.run,
            "/api/reset": session.reset,
            "/api/seek": session.seek,
            "/api/trace": session.trace,
            "/api/control": session.control,
            "/api/record": session.record,
        }
        action = actions.get(urlparse(self.path).path)
        if action is None:
            return self._json({"error": "not found"}, 404)
        return self._guarded(lambda: action(request))


def main():
    """Parse arguments and serve until interrupted."""
    parser = argparse.ArgumentParser(description="Serve a Graphion model, or a recording of one, to the spike UI.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", choices=("ber", "ddc"), default="ddc")
    source.add_argument("--replay", metavar="DIR", help="serve a recording; no model is built")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--trace", action="store_true", help="start with call tracing on")
    args = parser.parse_args()
    if args.replay:
        from spikes.inspection.recording import ReplayAdapter

        adapter = ReplayAdapter(args.replay)
        label = f"replay of {adapter.name}"
    else:
        from spikes.inspection.adapter import NumpyFrameAdapter, build_model

        adapter = NumpyFrameAdapter(build_model(args.model), args.model)
        adapter.set_trace(args.trace)
        label = f"model {args.model}"
    Handler.session = Session(adapter, record_root=HERE / "recordings")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Graphion inspection spike · {label} · http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
