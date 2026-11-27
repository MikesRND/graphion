"""Measure what attaching the inspection layer costs the model.

    python -m spikes.inspection.bench [--seconds 1.0] [--rounds 5]

Every figure is the model's speed under some condition divided by its bare
speed, where bare is the same model object run in a plain loop with the
adapter's thread parked. Three things keep that ratio honest:

- The pair is measured back to back on one model instance, reset to the same
  state first, because the examples do not run at a constant speed (the DDC
  testbench slows by about a third over its first 20,000 ticks on its own).
- Rounds interleave the conditions, and the median and the spread are reported.
- Clients are real ones: separate processes polling the real HTTP server, which
  runs in the model's process as it does in use.
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer

from spikes.inspection.adapter import NumpyFrameAdapter, build_model
from spikes.inspection.server import Handler
from spikes.inspection.session import Session

WARM_SECONDS = 0.2


def stream_probes(session, count=4):
    """Pick the first ``count`` distinct signals that have a rate: what a page of scopes would watch."""
    doc = session.adapter.document()
    out, seen = [], set()
    for path, facts in doc["facts"].items():
        for port, port_facts in facts["ports"].items():
            producer = port_facts.get("producer")
            if not port_facts.get("rate") or producer in seen:
                continue
            seen.add(producer)
            # ``view`` and ``max`` are what the layer before snapshots needed; today's ignores them.
            view = "spectrum" if len(out) % 2 else "time"
            out.append({"id": f"p{len(out)}", "path": path, "port": port, "view": view, "max": 512})
            if len(out) == count:
                return out
    return out


class Clients:
    """Some polling client processes, started before a measurement and stopped after it."""

    def __init__(self, address, count, hz, path, request):
        command = [sys.executable, "-m", "spikes.inspection.bench_client", address[0], str(address[1]), str(hz), path]
        self.processes = [
            subprocess.Popen([*command, json.dumps(request)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            for _ in range(count)
        ]
        for process in self.processes:
            process.stdout.readline()

    def stop(self):
        """Stop the clients and return each one's totals."""
        totals = []
        for process in self.processes:
            process.stdin.close()
            totals.append(json.loads(process.stdout.readline()))
            process.wait()
        return totals


def prepare(adapter):
    """Put the model in the same state before every measurement."""
    adapter.reset()
    adapter.advance("tick", 50)


def bare(adapter, seconds):
    """Ticks per second of the model alone, run on this thread while the adapter's thread is parked."""
    root = adapter.root
    prepare(adapter)
    deadline = time.perf_counter() + WARM_SECONDS
    while time.perf_counter() < deadline:
        root.run()
    count = 0
    started = time.perf_counter()
    deadline = started + seconds
    while time.perf_counter() < deadline:
        for _ in range(10):
            root.run()
        count += 10
    return count / (time.perf_counter() - started)


def attached(session, seconds, clients=None):
    """Ticks per second while the adapter free-runs the model, with the given clients polling."""
    adapter = session.adapter
    prepare(adapter)
    running = Clients(*clients) if clients else None
    session.run({"on": True})
    time.sleep(WARM_SECONDS)
    first = adapter.position()["tick"]
    started = time.perf_counter()
    time.sleep(seconds)
    last = adapter.position()["tick"]
    elapsed = time.perf_counter() - started
    totals = running.stop() if running else []
    session.run({"on": False})
    return (last - first) / elapsed, totals


def stepped(session, seconds, clients):
    """Ticks per second when a client asks for one tick at a time; the model only runs when asked."""
    adapter = session.adapter
    prepare(adapter)
    running = Clients(*clients)
    time.sleep(WARM_SECONDS)
    first = adapter.position()["tick"]
    started = time.perf_counter()
    time.sleep(seconds)
    last = adapter.position()["tick"]
    elapsed = time.perf_counter() - started
    return (last - first) / elapsed, running.stop()


def measure(name, seconds, rounds):
    """Return the labelled ratios for one model: {label: [(ratio, client totals), ...]}."""
    session = Session(NumpyFrameAdapter(build_model(name), name))
    Handler.session = session
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = server.server_address
    probes = stream_probes(session)
    watch = {"probes": probes, "inspect": probes[0]["path"]}

    def polling(count):
        return (address, count, 30, "/api/observe", watch)

    def traced(run):
        def wrapped():
            session.trace({"on": True})
            try:
                return run()
            finally:
                session.trace({"on": False})

        return wrapped

    def recording():
        with tempfile.TemporaryDirectory() as directory:
            session.record({"on": True, "path": directory})
            try:
                return attached(session, seconds)
            finally:
                session.record({"on": False})

    conditions = {
        "Attached, nobody watching": lambda: attached(session, seconds),
        f"{len(probes)} probes at 30 Hz, one client": lambda: attached(session, seconds, polling(1)),
        f"{len(probes)} probes at 30 Hz, three clients": lambda: attached(session, seconds, polling(3)),
        "Call tracing on, nobody watching": traced(lambda: attached(session, seconds)),
        f"Call tracing on, {len(probes)} probes at 30 Hz, one client": traced(
            lambda: attached(session, seconds, polling(1))
        ),
        "One client asks for one tick at a time": lambda: stepped(
            session, seconds, (address, 1, 0, "/api/advance", watch)
        ),
        "Recording every output": recording,
    }
    results = {label: [] for label in conditions}
    baselines = []
    for _ in range(rounds):
        for label, run in conditions.items():
            baseline = bare(session.adapter, seconds)
            rate, totals = run()
            baselines.append(baseline)
            results[label].append((rate / baseline, totals))
    session.adapter.close()
    server.shutdown()
    return results, baselines, len(session.adapter.blocks)


def main():
    """Print one table per example model."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["ber", "ddc"])
    args = parser.parse_args()
    for name in args.models:
        results, baselines, blocks = measure(name, args.seconds, args.rounds)
        base = statistics.median(baselines)
        print(
            f"\n{name}: {blocks} blocks, bare {base:.0f} ticks/s ({1e6 / base:.0f} µs per tick), "
            f"bare spread {100 * min(baselines) / base:.0f}..{100 * max(baselines) / base:.0f}% of its median; "
            f"median of {args.rounds} rounds of {args.seconds:g} s"
        )
        print(f"  {'condition':<52} {'of bare':>8} {'spread':>14} {'updates/s per client':>22} {'kB/update':>10}")
        for label, samples in results.items():
            ratios = [ratio for ratio, _ in samples]
            totals = [total for _, clients in samples for total in clients]
            updates = statistics.median(t["updates"] / t["seconds"] for t in totals) if totals else None
            size = statistics.median(t["bytes"] / max(1, t["updates"]) / 1024 for t in totals) if totals else None
            print(
                f"  {label:<52} {100 * statistics.median(ratios):7.1f}% "
                f"{100 * min(ratios):6.1f}..{100 * max(ratios):5.1f}% "
                f"{'' if updates is None else f'{updates:.1f}':>22} {'' if size is None else f'{size:.1f}':>10}"
            )


if __name__ == "__main__":
    main()
