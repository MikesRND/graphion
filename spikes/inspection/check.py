"""Checks for the inspection layer, run without a browser.

    python -m spikes.inspection.check

Each check prints PASS or FAIL; the process exits nonzero on any failure.
"""

import sys
import tempfile
import time

from spikes.inspection.adapter import NumpyFrameAdapter, Unsupported, build_model
from spikes.inspection.recording import ReplayAdapter
from spikes.inspection.session import Session

RESULTS = []


def report(name, ok, detail=""):
    """Print one result line and remember it."""
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")


def probes_for(session):
    """One probe per display hint plus one spectrum, so several views are exercised."""
    doc = session.adapter.document()
    out = []
    for path, facts in doc["facts"].items():
        if facts["display"]:
            out.append({"id": f"hint:{path}", "path": path, **facts["display"], "max": 128})
    first = next(path for path, facts in doc["facts"].items() if any(f.get("rate") for f in facts["ports"].values()))
    port = next(name for name, f in doc["facts"][first]["ports"].items() if f.get("rate"))
    out.append({"id": "spectrum", "path": first, "port": port, "view": "spectrum", "max": 64})
    return out


def comparable(observation):
    """Strip what legitimately differs between a live run and its replay."""
    probes = {key: {k: v for k, v in payload.items() if k != "stamp"} for key, payload in observation["probes"].items()}
    spans = [(s["path"], s["entry"], s["depth"]) for s in observation["trace"]]
    return probes, observation["live"], spans


def check_model(name):
    """Exercise tracing, call stepping, staleness, recording and replay on one model."""
    session = Session(NumpyFrameAdapter(build_model(name), name))
    adapter = session.adapter
    probes = probes_for(session)
    inspect_path = probes[0]["path"]
    request = {"probes": probes, "inspect": inspect_path}

    first = session.observe(request)
    report(
        f"{name}: outputs are marked unrun before the first tick", all(p.get("unrun") for p in first["probes"].values())
    )

    untraced = session.advance({"n": 3, **request})
    assumed = {p["stamp"]["basis"] for p in untraced["probes"].values()}
    report(
        f"{name}: tracing off gives no spans and assumed stamps",
        untraced["trace"] == [] and assumed == {"assumed"},
        f"tick {untraced['position']['tick']}",
    )

    try:
        adapter.advance("call")
        refused = False
    except Unsupported:
        refused = True
    report(f"{name}: call stepping is refused while tracing is off", refused)

    session.trace({"on": True})
    traced = session.advance(request)
    order = [(s["path"], s["entry"]) for s in traced["trace"]]
    observed = {p["stamp"]["basis"] for p in traced["probes"].values()}
    report(
        f"{name}: tracing on gives spans and observed stamps",
        len(order) > 3 and observed == {"observed"},
        f"{len(order)} calls",
    )

    # Step through one whole tick a call at a time and compare with the traced order.
    stops = []
    stale_seen = fresh_seen = False
    tick_before = traced["position"]["tick"]
    for _ in range(len(order) + 2):
        stop = session.advance({"mode": "call", **request})
        if stop["position"]["state"] != "paused":
            break
        at = stop["position"]["at"]
        stops.append((at["path"], at["entry"]))
        for payload in stop["probes"].values():
            stale_seen |= payload["stamp"]["tick"] == tick_before
            fresh_seen |= payload["stamp"]["tick"] == tick_before + 1
    report(f"{name}: call stepping stops before each call in traced order", stops == order[1:], f"{len(stops)} stops")
    report(f"{name}: a paused tick shows both stale and fresh outputs", stale_seen and fresh_seen)
    report(
        f"{name}: stepping past the last call completes the tick",
        stop["position"]["state"] == "idle" and stop["position"]["tick"] == tick_before + 1,
    )

    # Pause part-way, then reset: the tick must be finished before the reset.
    session.advance({"mode": "call"})
    session.advance({"mode": "call"})
    after_reset = session.reset(request)
    report(
        f"{name}: reset from a paused tick returns to tick zero",
        after_reset["position"]["tick"] == 0 and after_reset["position"]["state"] == "idle",
    )

    # Record a few ticks, then replay them through the same session code.
    with tempfile.TemporaryDirectory() as directory:
        session.record({"on": True, "path": directory})
        live = [comparable(session.advance(request)) for _ in range(5)]
        session.record({"on": False})
        replay = Session(ReplayAdapter(directory))
        replayed = [comparable(replay.advance(request)) for _ in range(5)]
        report(
            f"{name}: replay reproduces every live observation exactly",
            replayed == live,
            f"{len(live)} ticks, {len(probes)} probes",
        )
        replay.seek({"index": 2})
        report(f"{name}: seeking a recording returns that tick", comparable(replay.observe(request)) == live[1])
        ended = replay.advance({"n": 99})["position"]
        report(f"{name}: a recording ends where it was stopped", ended["state"] == "ended" and ended["index"] == 5)
        try:
            replay.control({"path": inspect_path, "name": "x", "value": 1})
            read_only = False
        except Unsupported:
            read_only = True
        report(f"{name}: a recording refuses writes", read_only)
        params = [
            (b["path"], n)
            for b in adapter.document()["description"]["blocks"]
            for n, v in b["parameters"].items()
            if v["kind"] == "summary"
        ]
        if params:
            path, param = params[0]
            report(
                f"{name}: configuration arrays replay too",
                replay.value(path, param) == session.value(path, param),
                f"{path}.{param}",
            )

    # Free running: the model keeps going while a reader polls it.
    session.trace({"on": False})
    session.run({"on": True})
    time.sleep(0.3)
    # Poll back to back: a reader in a tight loop must not starve the model.
    polled = [session.observe(request)["position"]["tick"] for _ in range(50)]
    session.run({"on": False})
    stopped = session.observe(request)["position"]
    gained = polled[-1] - polled[0]
    report(
        f"{name}: free run advances while being polled in a tight loop",
        gained >= 40 and stopped["state"] == "idle",
        f"{gained} ticks across 50 polls",
    )
    adapter.close()


def main():
    """Run the checks on both example models."""
    for name in ("ber", "ddc"):
        check_model(name)
    print(f"{sum(RESULTS)} of {len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
