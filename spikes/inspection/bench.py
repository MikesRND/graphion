"""Measure what attaching the inspection layer costs the model.

    python -m spikes.inspection.bench [--seconds 1.5]

Each row runs the same model for a fixed time and reports root iterations per
second. "Bare" is the model alone in a plain loop with nothing attached; every
other row is compared with it.
"""

import argparse
import json
import tempfile
import threading
import time

from spikes.inspection.adapter import NumpyFrameAdapter, build_model
from spikes.inspection.check import probes_for
from spikes.inspection.session import Session


def bare(name, seconds):
    """Ticks per second of the model alone: no adapter, no thread, no instrumentation."""
    root = build_model(name)
    for _ in range(50):
        root.run()
    count = 0
    started = time.perf_counter()
    deadline = started + seconds
    while time.perf_counter() < deadline:
        for _ in range(20):
            root.run()
        count += 20
    return count / (time.perf_counter() - started)


def free_run(session, seconds, poll_hz=0.0, request=None):
    """Ticks per second while free running, optionally polled like a UI would poll."""
    stop = threading.Event()

    def poll():
        while not stop.wait(1.0 / poll_hz):
            # Encoding is part of what a transport costs the process, so include it.
            json.dumps(session.observe(request), default=str)

    poller = threading.Thread(target=poll, daemon=True) if poll_hz else None
    session.run({"on": True})
    time.sleep(0.15)
    first = session.adapter.position()["tick"]
    started = time.perf_counter()
    if poller:
        poller.start()
    time.sleep(seconds)
    last = session.adapter.position()["tick"]
    elapsed = time.perf_counter() - started
    stop.set()
    if poller:
        poller.join()
    session.run({"on": False})
    return (last - first) / elapsed


def stepped(session, seconds, request):
    """Ticks per second when a client asks for one tick at a time and observes each."""
    count = 0
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        json.dumps(session.advance(request), default=str)
        count += 1
    return count / (time.perf_counter() - started)


def measure(name, seconds):
    """Return the labelled measurements for one model."""
    rows = [("Bare model, plain loop (headless)", bare(name, seconds))]
    session = Session(NumpyFrameAdapter(build_model(name), name))
    probes = probes_for(session)
    request = {"probes": probes, "inspect": probes[0]["path"]}
    session.advance({"n": 50})
    rows.append(("Attached, tracing off, nobody watching", free_run(session, seconds)))
    rows.append(("Attached, tracing off, UI polling at 15 Hz", free_run(session, seconds, 15, request)))
    session.trace({"on": True})
    rows.append(("Attached, call tracing on, nobody watching", free_run(session, seconds)))
    rows.append(("Attached, call tracing on, UI polling at 15 Hz", free_run(session, seconds, 15, request)))
    session.trace({"on": False})
    rows.append(("Attached, tracing off, client steps every tick", stepped(session, seconds, request)))
    with tempfile.TemporaryDirectory() as directory:
        session.record({"on": True, "path": directory})
        rows.append(("Attached, recording every output", free_run(session, seconds)))
        session.record({"on": False})
    session.adapter.close()
    return rows, len(probes), len(session.adapter.blocks)


def main():
    """Print one table per example model."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=1.5)
    args = parser.parse_args()
    for name in ("ber", "ddc"):
        rows, probes, blocks = measure(name, args.seconds)
        baseline = rows[0][1]
        print(f"\n{name}: {blocks} blocks, {probes} probes when polled, {1e6 / baseline:.0f} µs per tick bare")
        for label, rate in rows:
            extra = 1e6 / rate - 1e6 / baseline
            print(f"  {label:<50} {rate:8.0f} ticks/s  {100 * rate / baseline:5.1f}%  {extra:+8.1f} µs/tick")


if __name__ == "__main__":
    main()
