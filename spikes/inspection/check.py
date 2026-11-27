"""Checks for the inspection layer, run without a browser.

    python -m spikes.inspection.check

Each check prints PASS or FAIL; the process exits nonzero on any failure.
"""

import enum
import json
import sys
import tempfile
import time

import numpy as np

from spikes.inspection import adapter as live
from spikes.inspection import snapshots, views, wire
from spikes.inspection.adapter import NumpyFrameAdapter, Unsupported, build_model
from spikes.inspection.recording import ReplayAdapter
from spikes.inspection.session import Session

RESULTS = []


def report(name, ok, detail=""):
    """Print one result line and remember it."""
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")


def probes_for(session, limit=None):
    """One probe per display hint plus one more stream port, so several signals are exercised."""
    doc = session.adapter.document()
    out = []
    for path, facts in doc["facts"].items():
        if facts["display"]:
            out.append({"id": f"hint:{path}", "path": path, "port": facts["display"]["port"]})
    first = next(path for path, facts in doc["facts"].items() if any(f.get("rate") for f in facts["ports"].values()))
    port = next(name for name, f in doc["facts"][first]["ports"].items() if f.get("rate"))
    out.append({"id": "stream", "path": first, "port": port})
    return out[:limit]


def comparable(observation):
    """Strip what legitimately differs between a live run and its replay."""
    frames = {}
    for probe, meta in observation["frames"].items():
        data = observation.array(probe)
        frames[probe] = (
            {k: v for k, v in meta.items() if k != "stamp"},
            None if data is None else (data.dtype.str, data.tobytes()),
        )
    spans = [(s["path"], s["entry"], s["depth"]) for s in observation["trace"]]
    return frames, observation["live"], spans


def stamps(observation):
    """Return every frame's stamp, None for a frame without one."""
    return [meta.get("stamp") for meta in observation["frames"].values()]


def ticks_per_second(adapter, seconds, action):
    """Measure the model's speed while ``action`` is called back to back on this thread."""
    first = adapter.position()["tick"]
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        action()
    return (adapter.position()["tick"] - first) / (time.perf_counter() - started)


def check_model(name):
    """Exercise tracing, call stepping, staleness, recording and replay on one model."""
    session = Session(NumpyFrameAdapter(build_model(name), name))
    adapter = session.adapter
    probes = probes_for(session)
    inspect_path = probes[0]["path"]
    request = {"probes": probes, "inspect": inspect_path}

    first = session.observe(request)
    report(
        f"{name}: outputs are marked unrun before the first tick, and no samples are sent",
        all(meta.get("unrun") for meta in first["frames"].values()) and not first.arrays,
    )

    untraced = session.advance({"n": 3, **request})
    report(
        f"{name}: tracing off gives no spans and assumed stamps",
        untraced["trace"] == [] and {s["basis"] for s in stamps(untraced)} == {"assumed"},
        f"tick {untraced['position']['tick']}",
    )
    whole = all(
        meta["count"] == meta["total"] and not meta["truncated"] and untraced.array(probe).size == meta["count"]
        for probe, meta in untraced["frames"].items()
    )
    report(f"{name}: each frame arrives whole, with its dtype, shape and stamp", whole)

    report(
        f"{name}: asking again with the same serial returns nothing",
        session.observe({**request, "since": untraced["serial"]}) is None,
    )
    stepped = session.advance(request)
    newer = session.observe({**request, "since": untraced["serial"]})
    report(
        f"{name}: a client behind by one step is given the newer snapshot",
        newer is not None and newer["serial"] == stepped["serial"] and newer["position"]["tick"] == 4,
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
    report(
        f"{name}: tracing on gives spans and observed stamps",
        len(order) > 3 and {s["basis"] for s in stamps(traced)} == {"observed"},
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
        for stamp in stamps(stop):
            stale_seen |= stamp["tick"] == tick_before
            fresh_seen |= stamp["tick"] == tick_before + 1
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
        recorded = [comparable(session.advance(request)) for _ in range(5)]
        session.record({"on": False})
        replay = Session(ReplayAdapter(directory))
        replayed = [comparable(replay.advance(request)) for _ in range(5)]
        report(
            f"{name}: replay gives every frame back byte for byte",
            replayed == recorded,
            f"{len(recorded)} ticks, {len(probes)} probes",
        )
        replay.seek({"index": 2})
        report(f"{name}: seeking a recording returns that tick", comparable(replay.observe(request)) == recorded[1])
        seen = replay.observe(request)["serial"]
        report(
            f"{name}: a recording answers nothing new until its cursor moves",
            replay.observe({**request, "since": seen}) is None
            and replay.advance(request)["serial"] > seen
            and replay.observe({**request, "since": seen}) is not None,
        )
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

    # Free running: the model keeps going while readers ask as fast as they can.
    session.trace({"on": False})
    session.run({"on": True})
    time.sleep(0.3)
    unpolled = ticks_per_second(adapter, 0.4, lambda: time.sleep(0.01))
    blind = ticks_per_second(adapter, 0.4, lambda: session.observe(request))
    seen = [None]

    def follow():
        observation = session.observe({**request, "since": seen[0]})
        if observation is not None:
            seen[0] = observation["serial"]

    following = ticks_per_second(adapter, 0.4, follow)
    session.run({"on": False})
    stopped = session.observe(request)["position"]
    report(
        f"{name}: a client asking in a tight loop does not starve the model",
        min(blind, following) > 0.5 * unpolled and stopped["state"] == "idle",
        f"{100 * blind / unpolled:.0f}% of unpolled speed without since, {100 * following / unpolled:.0f}% with",
    )
    check_publishing(name, session, probes)
    adapter.close()


def check_publishing(name, session, probes):
    """Check what a running model is asked to copy, and when."""
    adapter = session.adapter
    one = {"probes": probes[:1]}
    key = session._key(probes[0])

    # Nobody watching: a running model copies nothing at all.
    session.observe(one)
    time.sleep(live.LEASE_SECONDS + 0.2)
    session.run({"on": True, "probes": []})
    # The request that started the run is itself an ask, and is owed one snapshot; let that pass.
    time.sleep(0.2)
    before = (adapter.copies, adapter._serial)
    time.sleep(0.5)
    idle_cost = (adapter.copies - before[0], adapter._serial - before[1])
    report(f"{name}: a running model nobody asks makes no snapshots and no copies", idle_cost == (0, 0), str(idle_cost))

    # Two clients on one port, polling as fast as they can: the rate cap bounds the copies.
    serials = set()
    tagged = set()
    before, started = adapter.copies, time.perf_counter()
    while time.perf_counter() - started < 1.0:
        for _client in range(2):
            observation = session.observe(one)
            serials.add(observation["serial"])
            tagged.update((s["tick"], observation["position"]["tick"]) for s in stamps(observation) if s)
    copied = adapter.copies - before
    cap = live.PUBLISH_HZ + 3
    report(
        f"{name}: two clients on one port cost one copy per published snapshot, at most the rate cap",
        10 <= len(serials) <= cap and copied <= cap,
        f"{len(serials)} snapshots and {copied} copies in one second",
    )
    report(
        f"{name}: a running frame's stamp names the tick its snapshot was made at",
        tagged and all(stamp == tick for stamp, tick in tagged),
        f"{len(tagged)} snapshots",
    )

    # A second port, watched by another client, joins the same snapshots.
    if len(probes) > 1:
        both = {"probes": probes[:2]}
        other = session._key(probes[1])
        deadline = time.perf_counter() + 1.0
        joined = False
        while time.perf_counter() < deadline and not joined:
            joined = len(session.observe(both)["frames"]) == 2
            time.sleep(0.01)
        report(f"{name}: a port asked for mid-run appears in a later snapshot", joined)
        # Stop asking for the first; keep asking for the second.
        deadline = time.perf_counter() + live.LEASE_SECONDS + 1.0
        while time.perf_counter() < deadline:
            session.observe({"probes": probes[1:2]})
            time.sleep(0.02)
        watched = set(adapter._latest.frames)
        report(
            f"{name}: a port nobody asks for any more drops out after its lease",
            key not in watched and other in watched,
            f"{len(watched)} still watched",
        )
    session.run({"on": False})


def check_unbound():
    """Attach a model whose external input nobody binds; it must be described, probed and reported as faulted."""
    try:
        session = Session(NumpyFrameAdapter(build_model("unbound"), "unbound"))
    except Exception as error:
        report("unbound: a model with an unbound input can be attached", False, f"{type(error).__name__}: {error}")
        return
    adapter = session.adapter
    facts = adapter.document()["facts"]["root"]["ports"]["x"]
    report(
        "unbound: an unbound input still has its size, from the description",
        facts["nbytes"] == 4096 * 8,
        f"{facts['nbytes']} bytes",
    )
    probes = [{"id": "in", "path": "root", "port": "x"}, {"id": "inner", "path": "root/tune", "port": "x"}]
    request = {"probes": probes, "inspect": "root/tune"}
    seen = session.observe(request)
    report(
        "unbound: probing the unbound input says nothing is bound, here and where it is forwarded",
        all("bound" in seen["frames"][p["id"]].get("missing", "") for p in probes),
        seen["frames"]["in"].get("missing", ""),
    )
    after = session.advance(request)
    position = after["position"]
    report(
        "unbound: running it is reported as a fault, not raised",
        position["state"] == "faulted" and "unset inputs" in (position["error"] or ""),
        position["error"] or "",
    )
    body = wire.encode(session.observe(request))
    report(
        "unbound: a faulted model still answers observations", wire.decode(body)[0]["position"]["state"] == "faulted"
    )
    cleared = session.reset(request)["position"]
    report("unbound: reset clears the fault", cleared["state"] == "idle" and cleared["error"] is None)
    adapter.close()


class _Mode(enum.Enum):
    FAST = 1


def check_values():
    """Values a strict JSON encoder would refuse must still summarize and encode."""
    cases = {
        "NaN": float("nan"),
        "infinity": float("-inf"),
        "NumPy NaN": np.float64("nan"),
        "complex": 1.5 - 2j,
        "NumPy complex": np.complex128(3j),
        "complex NaN": complex(float("nan"), 1.0),
        "enum": _Mode.FAST,
        "NaN array": np.array([np.nan, 1.0, np.inf]),
        "complex array": np.array([1 + 1j, np.nan]),
    }
    for label, value in cases.items():
        summary = views.summarize(value)
        try:
            json.dumps(summary, allow_nan=False)
            ok = summary["kind"] != "opaque"
        except ValueError:
            ok = False
        report(f"values: {label} summarizes to strict JSON", ok, str(summary.get("value", summary.get("head"))))

    long = np.arange(snapshots.MAX_SAMPLES + 10, dtype=np.float32).reshape(1, -1)
    frame = snapshots.take(long, 0, {"tick": 1, "basis": "assumed"})
    meta = frame.meta()
    report(
        "values: a channel over the cap is cut and says so, with its full shape",
        meta["truncated"] and meta["count"] == snapshots.MAX_SAMPLES and meta["shape"] == list(long.shape),
        f"{meta['count']} of {meta['total']}",
    )
    two = np.arange(8, dtype=np.complex128).reshape(2, 4)
    second = snapshots.take(two, 1, {"tick": 1, "basis": "assumed"})
    report(
        "values: a frame carries only the channel asked for, as its own copy",
        np.array_equal(second.data, two[1]) and not np.shares_memory(second.data, two),
    )
    report(
        "values: a channel the port does not have is reported missing", "channel" in snapshots.take(two, 5, {}).missing
    )


def main():
    """Run the checks on both example models and the awkward cases."""
    for name in ("ber", "ddc"):
        check_model(name)
    check_unbound()
    check_values()
    print(f"{sum(RESULTS)} of {len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
